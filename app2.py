from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load your trained model and scaler
model = tf.keras.models.load_model('models/model.h5')
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    crop_encoded = data['label_encoded']
    N = data['N']
    P = data['P']
    K = data['K']
    ph = data['ph']

    input_data = np.array([[crop_encoded, N, P, K, ph]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    return jsonify({
        'predicted_N': prediction[0][0],
        'predicted_P': prediction[0][1],
        'predicted_K': prediction[0][2],
        'predicted_pH': prediction[0][3]
    })

if __name__ == '__main__':
    app.run(debug=True)
