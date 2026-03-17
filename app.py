from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

with open('best_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # futuristic UI

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    prediction = model.predict([text])
    return jsonify({'sentiment': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
