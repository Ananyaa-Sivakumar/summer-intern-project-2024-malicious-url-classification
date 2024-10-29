from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model, tokenizer, label encoder, and scaler
with open('url_classification_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

max_len = 100  # This should match what was used during training

def predict_url(url):
    # Preprocess the new URL
    url_sequence = tokenizer.texts_to_sequences([url])
    url_sequence = pad_sequences(url_sequence, maxlen=max_len)

    # Feature engineering for the new URL
    url_length = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(not c.isalnum() for c in url)
    additional_features_new_url = scaler.transform([[url_length, num_digits, num_special_chars]])

    # Combine the URL sequence with the additional features
    url_combined = np.hstack((url_sequence, additional_features_new_url))

    # Make a prediction
    prediction = model.predict(url_combined)
    predicted_label = le.inverse_transform([np.argmax(prediction, axis=1)[0]])

    return predicted_label[0]

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url_input = data.get('url')

    if not url_input:
        return jsonify({'error': 'No URL provided'}), 400

    predicted_class = predict_url(url_input)  # Predict the class using the function
    return jsonify({'prediction': predicted_class})

# Define the route for the frontend form submission
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
