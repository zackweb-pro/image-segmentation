from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as T
import numpy as np

app = Flask(__name__)

# Load DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# COCO class for wall is 12 (for ADE20K, wall is 12, for COCO, wall is not present)
# DeepLabV3 pretrained on COCO does not have 'wall' class, but ADE20K does. We'll use 12 for wall in ADE20K.
WALL_CLASS = 12

# Preprocessing
transform = T.Compose([
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def segment_walls(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    # Mask for wall
    mask = (output_predictions == WALL_CLASS).astype(np.uint8) * 255
    return mask

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    mask = segment_walls(image)
    mask_img = Image.fromarray(mask)
    buf = io.BytesIO()
    mask_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
