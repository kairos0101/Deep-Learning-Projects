from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="/Users/jaydemirandilla/DenseNet121_WebApp/models/quantized_densenet121_62categories.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_path):
    """Preprocess the image for TFLite model inference."""
    target_size = input_details[0]['shape'][1:3]  # Get expected input size
    img = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB
    img = img.resize(target_size)  # Resize to model's input size
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image_path):
    """Perform inference on the image."""
    input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.squeeze(output_data)  # Return results as a flat array

class_indices_to_labels = {
            0:  'Adelfa',
            1:  'Akapulko',
            2:  'Alagaw',
            3:  'Alugbati',
            4:  'Ampalaya',
            5:  'Aratiles',
            6:  'Atis',
            7:  'Avocado',
            8:  'Balanoy',
            9: 'Banaba',
            10: 'Bawang',
            11: 'Bayabas',
            12: 'Begonia',
            13: 'Bignay',
            14: 'Chico',
            15: 'Dalandan',
            16: 'DamongMaria',
            17: 'Dayap',
            18: 'DragonFruit', 
            19: 'GarlicVines',
            20: 'Ginger',
            21: 'Gumamela',
            22: 'Guyabano',
            23: 'Ikmo',
            24: 'InsulinPlant',
            25: 'Ipil-ipil',
            26: 'Kalamansi',
            27: 'KalatsutsingPula',
            28: 'Kamantigi',
            29: 'Kamias',
            30: 'Kamote',
            31: 'KamotengKahoy',
            32: 'Lagikuay',
            33: 'Lagundi',
            34: 'LuyangDilaw',
            35: 'Madrecacao',
            36: 'Malunggay',
            37: 'Mango',
            38: 'Mani',
            39: 'Mayana',
            40: 'Mulberry',
            41: 'NeemTree',
            42: 'Niog-niogan',
            43: 'Oregano',
            44: 'Pako',
            45: 'Pandan-mabango',
            46: 'Pansit-pansitan',
            47: 'Sabila',
            48: 'Saluyot',
            49: 'Sambong',
            50: 'Sampa-sampalukan',
            51: 'Sampalok',
            52: 'Serpentina',
            53: 'SilingLabuyo',
            54: 'Suha',
            55: 'Takip-kohol',
            56: 'Tambalisa',
            57: 'Tanglad',
            58: 'Tawa-tawa',
            59: 'TsaangGubat',
            60: 'TubangBakod',
            61: 'YerbaBuena'
        }

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform prediction
        predictions = predict(file_path)

        # Top-3 predictions
        top_indices = predictions.argsort()[-3:][::-1]
        top_scores = predictions[top_indices]
        top_labels = [class_indices_to_labels[i] for i in top_indices]  # Replace with actual labels if available

        results = [{"label": label, "score": score} for label, score in zip(top_labels, top_scores)]
        
        return render_template('result.html', image_url=file_path, results=results)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
