import streamlit as st
import pandas as pd
import requests
from pathlib import Path

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide",
)


# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HOTELS_PATH = PROJECT_ROOT / "travel_capstone_dataset" / "hotels.csv"

FLIGHTS_PATH = PROJECT_ROOT / "travel_capstone_dataset" / "flights.csv"

FLASK_API_URL = "http://localhost:5001"


# ==========================================================
# LOAD HOTEL DATA
# ==========================================================


@st.cache_data
def load_hotels():

    df = pd.read_csv(HOTELS_PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        format="%m/%d/%Y",
    )

    return df


# ==========================================================
# LOAD FLIGHT DATA
# ==========================================================


@st.cache_data
def load_flights():

    df = pd.read_csv(FLIGHTS_PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        format="%m/%d/%Y",
    )

    return df


try:

    hotels = load_hotels()
    flights = load_flights()

except Exception as e:

    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ==========================================================
# HOTEL RECOMMENDATION ENGINE
# ==========================================================


def recommend_hotels(
    destination,
    days,
    max_budget,
    top_n=5,
):

    candidates = hotels[hotels["place"].eq(destination)].copy()

    candidates = candidates[candidates["total"] <= max_budget]

    if candidates.empty:

        return pd.DataFrame()

    candidates["days_difference"] = (candidates["days"] - days).abs()

    recommendations = (
        candidates.sort_values(
            by=[
                "days_difference",
                "total",
                "price",
            ]
        )
        .drop_duplicates(
            subset=["name"],
            keep="first",
        )
        .head(top_n)
        .copy()
    )

    return recommendations[
        [
            "name",
            "place",
            "days",
            "price",
            "total",
            "date",
        ]
    ]


# ==========================================================
# FLIGHT PREDICTION API
# ==========================================================


def predict_flight_price(data):

    try:

        response = requests.post(
            f"{FLASK_API_URL}/predict",
            json=data,
            timeout=30,
        )

        if response.status_code == 200:

            return response.json(), None

        return None, (
            f"API returned status " f"{response.status_code}: " f"{response.text}"
        )

    except requests.exceptions.ConnectionError:

        return None, (
            "Unable to connect to Flask API. "
            "Make sure Kubernetes port-forward "
            "is running on port 5001."
        )

    except requests.exceptions.Timeout:

        return None, "Request timed out."

    except Exception as e:

        return None, str(e)


# ==========================================================
# MAIN HEADER
# ==========================================================

st.title("✈️ Voyage Analytics")

st.write(
    "Travel analytics platform for "
    "hotel recommendations and flight "
    "price prediction."
)


# ==========================================================
# TABS
# ==========================================================

hotel_tab, flight_tab = st.tabs(
    [
        "🏨 Hotel Recommendation",
        "✈️ Flight Price Prediction",
    ]
)


# ==========================================================
# HOTEL RECOMMENDATION
# ==========================================================

with hotel_tab:

    st.subheader("🏨 Travel & Hotel Recommendation")

    st.write(
        "Find suitable hotel packages based on "
        "destination, trip duration, and budget."
    )

    # ------------------------------------------------------
    # HOTEL INPUTS
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    destinations = sorted(hotels["place"].dropna().unique())

    available_days = sorted(hotels["days"].dropna().unique())

    with col1:

        destination = st.selectbox(
            "Destination",
            destinations,
        )

    with col2:

        days = st.selectbox(
            "Number of Days",
            available_days,
        )

    with col3:

        max_total = float(hotels["total"].max())

        max_budget = st.number_input(
            "Maximum Budget ($)",
            min_value=100.0,
            max_value=max_total,
            value=min(
                750.0,
                max_total,
            ),
            step=50.0,
        )

    with col4:

        top_n = st.slider(
            "Recommendations",
            min_value=1,
            max_value=9,
            value=5,
        )

    st.divider()

    # ------------------------------------------------------
    # HOTEL RECOMMENDATION
    # ------------------------------------------------------

    if st.button(
        "🔎 Get Hotel Recommendations",
        type="primary",
    ):

        recommendations = recommend_hotels(
            destination=destination,
            days=days,
            max_budget=max_budget,
            top_n=top_n,
        )

        if recommendations.empty:

            st.warning(
                "No matching hotels found. "
                "Try increasing your budget "
                "or changing the number of days."
            )

        else:

            st.success(
                f"Found {len(recommendations)} "
                f"hotel recommendation"
                f"{'s' if len(recommendations) != 1 else ''}."
            )

            # --------------------------------------------------
            # HOTEL METRICS
            # --------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Lowest Total Cost",
                    f"${recommendations['total'].min():,.2f}",
                )

            with col2:

                st.metric(
                    "Average Total Cost",
                    f"${recommendations['total'].mean():,.2f}",
                )

            with col3:

                st.metric(
                    "Hotels Recommended",
                    len(recommendations),
                )

            # --------------------------------------------------
            # COST CHART
            # --------------------------------------------------

            st.subheader("💰 Total Cost Comparison")

            chart_data = recommendations.set_index("name")["total"]

            st.bar_chart(chart_data)

            # --------------------------------------------------
            # RECOMMENDED HOTELS
            # --------------------------------------------------

            st.subheader("🏨 Recommended Hotels")

            display_df = recommendations.copy()

            display_df["days"] = display_df["days"].astype(int)

            display_df["price"] = display_df["price"].apply(lambda x: f"${x:,.2f}")

            display_df["total"] = display_df["total"].apply(lambda x: f"${x:,.2f}")

            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

            display_df = display_df.rename(
                columns={
                    "name": "Hotel",
                    "place": "Destination",
                    "days": "Days",
                    "price": "Price / Day",
                    "total": "Total Cost",
                    "date": "Date",
                }
            )

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
            )


# ==========================================================
# FLIGHT PRICE PREDICTION
# ==========================================================

with flight_tab:

    st.subheader("✈️ Flight Price Prediction")

    st.write(
        "Predict flight prices using the " "MLflow-managed machine learning model."
    )

    # ------------------------------------------------------
    # FLIGHT ORIGIN OPTIONS
    # ------------------------------------------------------

    flight_from_options = sorted(flights["from"].dropna().astype(str).unique())

    default_from = (
        "Florianopolis (SC)"
        if "Florianopolis (SC)" in flight_from_options
        else flight_from_options[0]
    )

    flight_from = st.selectbox(
        "From",
        flight_from_options,
        index=flight_from_options.index(default_from),
    )

    # ------------------------------------------------------
    # DESTINATION OPTIONS BASED ON ORIGIN
    # ------------------------------------------------------

    route_data = flights[flights["from"].astype(str).eq(flight_from)].copy()

    flight_to_options = sorted(route_data["to"].dropna().astype(str).unique())

    if not flight_to_options:

        st.error("No destinations are available " f"for {flight_from}.")

        st.stop()

    default_to = (
        "Aracaju (SE)" if "Aracaju (SE)" in flight_to_options else flight_to_options[0]
    )

    flight_to = st.selectbox(
        "To",
        flight_to_options,
        index=flight_to_options.index(default_to),
    )

    # ------------------------------------------------------
    # ROUTE DATA
    # ------------------------------------------------------

    selected_route = flights[
        (flights["from"].astype(str) == flight_from)
        & (flights["to"].astype(str) == flight_to)
    ].copy()

    # ------------------------------------------------------
    # FLIGHT TYPE OPTIONS
    # ------------------------------------------------------

    flight_type_options = sorted(
        selected_route["flightType"].dropna().astype(str).unique()
    )

    default_flight_type = (
        "firstClass" if "firstClass" in flight_type_options else flight_type_options[0]
    )

    flight_type = st.selectbox(
        "Flight Type",
        flight_type_options,
        index=flight_type_options.index(default_flight_type),
    )

    # ------------------------------------------------------
    # AGENCY OPTIONS
    # ------------------------------------------------------

    agency_data = selected_route[
        selected_route["flightType"].astype(str).eq(flight_type)
    ]

    agency_options = sorted(agency_data["agency"].dropna().astype(str).unique())

    default_agency = "CloudFy" if "CloudFy" in agency_options else agency_options[0]

    agency = st.selectbox(
        "Agency",
        agency_options,
        index=agency_options.index(default_agency),
    )

    # ------------------------------------------------------
    # NUMERICAL DEFAULTS
    # ------------------------------------------------------

    combination_data = agency_data[agency_data["agency"].astype(str).eq(agency)]

    if combination_data.empty:

        combination_data = selected_route

    default_time = float(combination_data["time"].median())

    default_distance = float(combination_data["distance"].median())

    # ------------------------------------------------------
    # NUMERICAL INPUTS
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        flight_time = st.number_input(
            "Flight Time",
            min_value=0.0,
            value=round(
                default_time,
                2,
            ),
            step=10.0,
        )

    with col2:

        distance = st.number_input(
            "Distance",
            min_value=0.0,
            value=round(
                default_distance,
                2,
            ),
            step=50.0,
        )

    flight_date = st.date_input("Travel Date")

    st.divider()

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    if st.button(
        "💰 Predict Flight Price",
        type="primary",
    ):

        if flight_time <= 0:

            st.error("Flight time must be greater than 0.")

        elif distance <= 0:

            st.error("Distance must be greater than 0.")

        else:

            prediction_data = {
                "from": flight_from,
                "to": flight_to,
                "flightType": flight_type,
                "time": flight_time,
                "distance": distance,
                "agency": agency,
                "date": flight_date.strftime("%Y-%m-%d"),
            }

            with st.spinner("Predicting flight price..."):

                result, error = predict_flight_price(prediction_data)

            if error:

                st.error(error)

            else:

                predicted_price = float(result["predicted_price"])

                st.success("Flight price prediction successful!")

                st.metric(
                    "✈️ Predicted Flight Price",
                    f"${predicted_price:,.2f}",
                )

                st.caption(
                    f"Model: {result['model']}  |  " f"Alias: {result['model_alias']}"
                )


# ==========================================================
# DATASET INFORMATION
# ==========================================================

st.divider()

with st.expander("📊 Dataset Information"):

    st.markdown("### 🏨 Hotel Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Hotel Records",
            f"{len(hotels):,}",
        )

    with col2:

        st.metric(
            "Hotel Destinations",
            hotels["place"].nunique(),
        )

    with col3:

        st.metric(
            "Hotels",
            hotels["name"].nunique(),
        )

    st.divider()

    st.markdown("### ✈️ Flight Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Flight Records",
            f"{len(flights):,}",
        )

    with col2:

        st.metric(
            "Origin Cities",
            flights["from"].nunique(),
        )

    with col3:

        st.metric(
            "Destination Cities",
            flights["to"].nunique(),
        )

    with col4:

        st.metric(
            "Agencies",
            flights["agency"].nunique(),
        )
