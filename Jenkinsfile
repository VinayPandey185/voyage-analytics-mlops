pipeline {
    agent any

    environment {
        IMAGE_NAME = "travel-mlops-flask"
        IMAGE_TAG  = "ci"
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking project source...'
                checkout scm
            }
        }

        stage('Validate Project') {
            steps {
                echo 'Validating project structure...'

                bat 'python --version'
                bat 'docker --version'

                bat 'if not exist app\\app.py exit /b 1'
                bat 'if not exist docker\\Dockerfile exit /b 1'
                bat 'if not exist kubernetes\\deployment.yaml exit /b 1'
                bat 'if not exist models\\flight_price_pipeline.joblib exit /b 1'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -f docker\\Dockerfile -t %IMAGE_NAME%:%IMAGE_TAG% .'
            }
        }

        stage('Test Container Image') {
            steps {
                bat 'docker run --rm %IMAGE_NAME%:%IMAGE_TAG% python -c "import mlflow, lightgbm, sklearn; print(\'Container dependencies OK\')"'
            }
        }

        stage('Deployment Validation') {
            steps {
                echo 'Kubernetes manifests are present and ready.'
                bat 'kubectl version --client'
                bat 'kubectl apply --dry-run=client -f kubernetes\\deployment.yaml'
                bat 'kubectl apply --dry-run=client -f kubernetes\\service.yaml'
            }
        }
    }

    post {
        success {
            echo 'CI/CD pipeline completed successfully.'
        }

        failure {
            echo 'CI/CD pipeline failed. Check the stage logs.'
        }
    }
}