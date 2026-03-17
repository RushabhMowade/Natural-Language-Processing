# 📊 Sentiment Analysis API

A machine learning-powered web service that classifies text sentiment as **Positive** or **Negative**. This project bridges the gap between a data science research notebook and a production-ready API.

## 🚀 Live Demo
The API is currently hosted on Render:
🔗 [https://my-sentiment-tool.onrender.com](https://my-sentiment-tool.onrender.com)

## 🧠 Project Overview
This project uses Natural Language Processing (NLP) to analyze the sentiment of user-provided text. The workflow involved training a model in a Jupyter environment and deploying it using a Flask wrapper.

* **Model:** Scikit-Learn (Trained and exported as a pickle file)
* **API Framework:** Flask (Python)
* **Web Server:** Gunicorn
* **Deployment:** Render (Native Python Runtime)
* **Training Source:** `NLP.ipynb`

## 📁 Repository Structure
* `app.py`: The Flask application script that loads the model and serves the API endpoints.
* `best_sentiment_model.pkl`: The serialized pre-trained machine learning model.
* `requirements.txt`: A list of all Python dependencies (Flask, Scikit-Learn, Gunicorn, etc.).
* `NLP.ipynb`: The original Jupyter Notebook containing data exploration, preprocessing, and model training.

## 🛠️ Local Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RushabhMowade/Natural-Language-Processing.git](https://github.com/RushabhMowade/Natural-Language-Processing.git)
    cd Natural-Language-Processing
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the local server:**
    ```bash
    python app.py
    ```

## 🔌 API Usage

You can test the sentiment analysis by sending a **POST** request to the `/predict` endpoint.

**Request Format:**
- **URL:** `https://my-sentiment-tool.onrender.com/predict`
- **Method:** `POST`
- **Body (JSON):** ```json
  {
    "text": "The project deployment was successful and I am very happy!"
  }
