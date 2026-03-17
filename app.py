import os
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global model variable
model = None

@app.before_first_request
def load_model():
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'best_sentiment_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded on Render!")
        else:
            print("❌ Model file not found!")
    except Exception as e:
        print(f"❌ Model error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not ready'}), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        prediction = model.predict([text])
        return jsonify({'sentiment': str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
