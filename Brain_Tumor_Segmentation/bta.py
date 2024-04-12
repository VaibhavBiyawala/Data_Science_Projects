from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load the UNet model
model = tf.saved_model.load("unet_model.pl")

# Define a function to process uploaded images
def process_image(image):
    # Preprocess the image (resize, normalize, etc.)
    # Here you should preprocess the image according to the requirements of your model
    # For example, if your model expects input images of size (256, 256, 3), you can do:
    image = image.resize((128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    # Predict using the model
    output = model.predict(image)
    # Postprocess the output image if necessary
    # For example, convert it to PIL Image
    output_image = Image.fromarray((output[0] * 255).astype('uint8'))
    return output_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the uploaded image
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        # Process the uploaded image
        input_image = Image.open(image_path)
        output_image = process_image(input_image)
        # Save the output image
        output_image_path = os.path.join('static', 'output_image.jpg')
        output_image.save(output_image_path)
        # Return the output image
        return redirect(url_for('result'))
    return redirect(request.url)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run()
