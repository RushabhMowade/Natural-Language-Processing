import streamlit as st
import os
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📝")

# --- 1. Model Loading (Replaces @app.before_first_request) ---
# We use @st.cache_resource so the model only loads once and stays in memory
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'best_sentiment_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.error("❌ Model file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        return None

model = load_model()

# --- 2. UI Layout (Replaces index.html) ---
st.title("Sentiment Analysis App")
st.write("Enter text below to predict its sentiment.")

# Text input widget
user_input = st.text_area("Input Text", placeholder="Type something here...")

# --- 3. Prediction Logic (Replaces @app.route('/predict')) ---
if st.button("Analyze Sentiment"):
    if model is None:
        st.warning("Model is not ready. Please check if the .pkl file exists.")
    elif not user_input.strip():
        st.info("Please enter some text first!")
    else:
        try:
            # Predict
            prediction = model.predict([user_input])
            sentiment = str(prediction[0])
            
            # Display result with some styling
            st.subheader("Result:")
            if sentiment.lower() == 'positive': # Adjust based on your model's labels
                st.success(f"The sentiment is: **{sentiment}**")
            elif sentiment.lower() == 'negative':
                st.error(f"The sentiment is: **{sentiment}**")
            else:
                st.info(f"The sentiment is: **{sentiment}**")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Footer (Optional)
st.divider()
st.caption("Powered by Streamlit & Scikit-Learn")
