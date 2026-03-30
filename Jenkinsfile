pipeline {
    agent any

    environment {
        AWS_ACCESS_KEY_ID = credentials('aws-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-key')
        AWS_SESSION_TOKEN = credentials('aws-session-token')

        DOCKER_USERNAME = credentials('docker-username')
        DOCKER_PASSWORD = credentials('docker-password')

        MLFLOW_TRACKING_URI = "http://host.docker.internal:8080"
        BUCKET_NAME = "2022bcs0010-mlops-assignment"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git 'https://github.com/<your-username>/<your-repo>.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                pip install --upgrade pip
                pip install -r requirements.txt
                pip install mlflow boto3 dvc[s3] scikit-learn
                '''
            }
        }

        stage('Configure AWS') {
            steps {
                sh '''
                aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
                aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
                aws configure set aws_session_token $AWS_SESSION_TOKEN
                aws configure set default.region us-east-1
                '''
            }
        }

        stage('Pull Data (DVC)') {
            steps {
                sh '''
                dvc pull
                '''
            }
        }

        stage('Train Model + MLflow Logging') {
            steps {
                sh '''
                export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
                python src/train.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $DOCKER_USERNAME/2022bcs0010-mlops-assignment .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                sh '''
                echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
                docker push $DOCKER_USERNAME/2022bcs0010-mlops-assignment
                '''
            }
        }
    }

    post {
        success {
            echo "Pipeline completed successfully!"
        }
        failure {
            echo "Pipeline failed!"
        }
    }
}