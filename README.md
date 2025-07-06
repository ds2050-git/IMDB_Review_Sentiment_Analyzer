# IMDB Review Sentiment Analyzer

## Overview
A sentiment analysis tool for IMDB movie reviews using a Simple RNN model built with TensorFlow. Includes a Streamlit web app for interactive review classification.

## Dependencies
- Python 3.8+
- Required packages (listed in `requirements.txt`):
  ```
  tensorflow==2.17.0
  pandas
  numpy
  scikit-learn
  tensorboard
  matplotlib
  streamlit
  ipykernel
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ds2050-git/IMDB_Review_Sentiment_Analyzer.git
   cd IMDB_Review_Sentiment_Analyzer
   ```
2. Set up a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model to generate `simple_rnn_imdb_review.h5`:
   ```bash
   python app.py
   ```

## Running the Project
- **Train the Model**: Run `app.py` to train and save the RNN model:
  ```bash
  python app.py
  ```
- **Test Predictions**: Run `Predictions.py` to classify a sample review:
  ```bash
  python Predictions.py
  ```
- **Launch Streamlit App**: Start the interactive web app:
  ```bash
  streamlit run Streamlit_App.py
  ```
  Open browser, enter a review, and click "Classify".
