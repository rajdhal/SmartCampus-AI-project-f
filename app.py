from flask import Flask, render_template, request, redirect, url_for, make_response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import io
from image_recognition import crop
import base64

app = Flask(__name__)

# Configure the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the YOLO model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty part without a filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Read the uploaded image using OpenCV
            file_data = file.read()
            nparr = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            
            # Crop the uploaded image based on the best bounding box using OpenCV
            im_cropped = crop(image)
            
            # Encode the cropped image to send as a response
            _, buffer = cv2.imencode('.jpg', im_cropped)
            cropped_image_data = base64.b64encode(buffer).decode('utf-8')
            
            return render_template('result.html', result_image=cropped_image_data)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)