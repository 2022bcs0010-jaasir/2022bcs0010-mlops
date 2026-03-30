pipeline {
    agent any

    environment {
        AWS_ACCESS_KEY_ID = credentials('aws-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-key')
        AWS_SESSION_TOKEN = credentials('aws-session-token')

        DOCKER_CREDS = credentials('docker-access')

        MLFLOW_TRACKING_URI = "http://host.docker.internal:8080"
        BUCKET_NAME = "2022bcs0010-mlops-assignment"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/2022bcs0010-jaasir/2022bcs0010-mlops'
            }
        }

       stage('Install Dependencies') {
            steps {
                sh '''
                # Create a virtual environment
                python3 -m venv venv

                # Activate it
                . venv/bin/activate

                # Upgrade pip inside venv only
                pip install --upgrade pip

                # Install project requirements
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
                MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI python src/train.py
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
                echo $DOCKER_CREDS_PSW | docker login -u $DOCKER_CREDS_USR --password-stdin
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