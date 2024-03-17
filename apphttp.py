from flask import Flask, request, jsonify, Response
import requests
from PIL import Image
from io import BytesIO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from settings import HTTP_PORT

app = Flask(__name__)

model = None
HTTP_INFERENCE_COUNT = Counter('app_http_inference_count', 'HTTP Inference Request Count')

@app.route('/predict', methods=['POST'])
def predict():
    HTTP_INFERENCE_COUNT.inc()

    data = request.get_json()
    image_url = data['url']
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    predictions = model(transform(image).unsqueeze(0))
    labels = predictions[0]['labels']
    object_names = [FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories'][label.item()]
                    for label in labels]
    response_data = {
        "objects": object_names
    }
    return jsonify(response_data), 200


@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    app.run(port=HTTP_PORT)
