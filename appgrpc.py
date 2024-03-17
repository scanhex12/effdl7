from concurrent import futures
import grpc
import time

import inference_pb2_grpc as inference_grpc
import inference_pb2 as inference

from settings import GRPC_PORT

from PIL import Image
from io import BytesIO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms
import requests

class InstanceDetectorService(inference_grpc.InstanceDetectorServicer):
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def Predict(self, request, context):
        image_url = request. url
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        predictions = self.model(transform(image).unsqueeze(0))
        labels = predictions[0]['labels']
        object_names = [FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories'][label.item()]
                    for label in labels]
        response_data = {
            "objects": object_names
        }
        return inference.InstanceDetectorOutput(objects=object_names)

if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_grpc.add_InstanceDetectorServicer_to_server(InstanceDetectorService(), server)
    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
