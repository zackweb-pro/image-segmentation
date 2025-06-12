from flask import Flask, request, send_file, jsonify, render_template
from PIL import Image
import io
import torch
import numpy as np
import os.path

# Import our custom wall segmentation module
from wall_segmentation import load_wall_segmentation_model, segment_walls

app = Flask(__name__)

# Path to the pretrained model weights
ENCODER_PATH = os.path.join('model_weights', 'transfer_encoder.pth')
DECODER_PATH = os.path.join('model_weights', 'transfer_decoder.pth')

# Load the wall segmentation model
if os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH):
    print(f"Loading wall segmentation model from {ENCODER_PATH} and {DECODER_PATH}")
    segmentation_model = load_wall_segmentation_model(ENCODER_PATH, DECODER_PATH)
    
    # Function that uses our loaded model
    def segment_walls_image(image):
        from wall_segmentation import segment_walls as segment_function
        mask = segment_function(segmentation_model, image)
        return mask
else:
    # If model weights are not found, print message and use dummy function
    print(f"WARNING: Model weights not found at {ENCODER_PATH} or {DECODER_PATH}")
    print("Please download model weights from: https://drive.google.com/drive/folders/1xh-MBuALwvNNFnLe-eofZU_wn8y3ZxJg")
    print("Download the 'Transfer learning - entire decoder' folder and place the weights in the model_weights directory")
    
    def segment_walls_image(image):
        # Return empty mask if no model is available
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    
    # Get original dimensions
    width, height = image.size
    
    # Resize large images to improve performance (max dimension of 1024 pixels)
    max_dim = 1024
    if width > max_dim or height > max_dim:
        # Calculate new dimensions while preserving aspect ratio
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        
        # Resize the image
        print(f"Resizing image from {width}x{height} to {new_width}x{new_height} for faster processing")
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Process the image
    mask = segment_walls_image(image)
    
    # Create output image
    mask_img = Image.fromarray(mask)
    
    # Return the segmentation mask
    buf = io.BytesIO()
    mask_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
