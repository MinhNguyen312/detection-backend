from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from sys import platform
from PIL import Image
from contextlib import asynccontextmanager
import torch
import io

#Uncomment if running on Windows
if platform == "win32":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


# On startup, load the model
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                             path=PATH, 
                             force_reload=True,
                             device='cpu' if not torch.cuda.is_available() else 'cuda')
        model.conf = 0.25
        model.iou = 0.45
        print(f"Model loaded successfully from {PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield

app = FastAPI(lifespan=lifespan)

origins= ["http://localhost","http://localhost:8080", "http://localhost:3000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/status/v1/health')
def check_health():
    return dict(msg='OK')


PATH = './model/best.pt'
model = None


@app.post('/upload')
async def upload(file: bytes = File(...)):

    global model
    if model is None:
        return {"error":"Model not initialized"}
    
    results = model(Image.open(io.BytesIO(file)))

    if platform == "win32":
        pathlib.PosixPath = temp         

    json_results = results_to_json(results, model)

    return json_results


def results_to_json(results, model):
    return [
        [
          {
          "class": int(pred[5]),
          "class_name": model.model.names[int(pred[5])],
          "bbox": [
                float(((pred[0] + pred[2]) / 2) / 416),  # xcenter
                float(((pred[1] + pred[3]) / 2) /416),  # ycenter
                float((pred[2] - pred[0])/416),        # width
                float((pred[3] - pred[1])/416)       # height
                ],
          "confidence": float(pred[4]),
          }
        for pred in result
        ]
      for result in results.xyxy
      ]
    

