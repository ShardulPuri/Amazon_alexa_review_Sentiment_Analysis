# Sentiment Analysis Web Application

## Overview
This project is a web application built with Flask that performs sentiment analysis on text input. It uses a machine learning model trained on the Amazon Alexa Reviews dataset to classify reviews as **Positive** or **Negative**.

Users can input single text phrases or upload CSV files with multiple sentences for batch predictions. The app returns sentiment classification along with a distribution pie chart for batch inputs.

## Features
- **Real-time sentiment prediction** on text input
- **Bulk prediction** via CSV file upload
- **Visual pie chart** to show sentiment distribution
- **Smooth and responsive UI** using Flask and Tailwind CSS
- **Pretrained XGBoost model** with optimized vectorizer and scaler

## Dataset
- Dataset: [Amazon Alexa Reviews (Kaggle)](https://www.kaggle.com/sid321axn/amazon-alexa-reviews)
- Dataset includes verified reviews and feedback labels used for binary sentiment classification.

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:https://github.com/ShardulPuri/Amazon_alexa_review_Sentiment_Analysis.git

- cd sentiment-flask-app

2. Create and activate a Python virtual environment (recommended):
- python -m venv venv
- source venv/bin/activate # Linux/macOS
- venv\Scripts\activate # Windows

3. Install dependencies:
- pip install -r requirements.txt

4. Run the Flask app:
- python app.py


### Access
Open your browser at `http://127.0.0.1:5000` to use the app.

## Project Structure
sentiment-flask-app/
├── app.py # Flask backend API
├── Models/
│ ├── model_xgb.pkl # Trained XGBoost model file
│ ├── scaler.pkl # MinMaxScaler for feature scaling
│ └── countVectorizer.pkl # CountVectorizer for text vectorization
├── templates/
│ ├── landing.html # Main HTML page
├── static/
│ └── (CSS, JS files if any)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## How It Works
- User submits text or CSV file for prediction.
- Flask backend loads pretrained model, scaler, and vectorizer.
- Text preprocessing includes cleaning, lowercasing, optional stemming, and vectorization.
- Model predicts sentiment; batch inputs return CSV with results and sentiment distribution chart.
- Prediction results and chart are rendered on the frontend.

## Model and Training Details
- Model: XGBoost classifier trained on cleaned reviews.
- Preprocessing: Text cleaned using regex, stopwords removed, Porter stemming applied.
- Features: Bag-of-Words created via CountVectorizer capped at 2500 features.
- Scaling: MinMaxScaler normalizes features.
- Performance: ~94% accuracy on held-out test data.

## Future Improvements
- Improve model with TF-IDF or word embeddings.
- Add support for multi-class sentiment ratings.
- Enhance UI with more interactive visualizations.
- Deploy to cloud platform for scalability.

## License
MIT License


