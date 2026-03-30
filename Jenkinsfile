pipeline {
    agent any

    environment {
        AWS_ACCESS_KEY_ID = credentials('aws-access-key')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-key')
        AWS_SESSION_TOKEN = credentials('aws-session-token')

        DOCKER_ACCESS = credentials('docker-access')
        MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"
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
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install "mlflow==2.13.0" boto3 "dvc[s3]" scikit-learn
                pip install -r requirements.txt
                '''
            }
        }

        stage('Configure AWS') {
            steps {
                sh '''
                . venv/bin/activate
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
                . venv/bin/activate
                dvc pull
                '''
            }
        }

        stage('Train Model + MLflow Logging') {
            steps {
                sh '''
                . venv/bin/activate
                export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
                python src/train.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $DOCKER_ACCESS/2022bcs0010-mlops-assignment .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                sh '''
                echo $DOCKER_ACCESS_PSW | docker login -u $DOCKER_ACCESS_USR --password-stdin
                docker push $DOCKER_ACCESS/2022bcs0010-mlops-assignment
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