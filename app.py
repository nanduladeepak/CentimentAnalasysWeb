from flask import Flask, render_template, request, jsonify
import librosa
import cv2
import os
from data_process import getPreProcessedData
from model import Predictor

centimentPrigictor = Predictor()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        inputVal = getPreProcessedData(file_path)
        print('pridicting')
        sentiment = centimentPrigictor.predict(inputVal)
        
        # You can pass audio and cap to your function for further processing
        # Example: your_function(audio, cap)

        return jsonify({'success': True, 'sentiment': sentiment.tolist()})  # Convert sentiment to list for JSON serialization

if __name__ == '__main__':
    app.run(debug=True)
