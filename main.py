from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import pathlib


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

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

@app.post('/upload')
async def upload(file: bytes = File(...)):

    PATH = './model/best.pt'
    model = None
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH, force_reload=False,
                               device='cpu' if not torch.cuda.is_available() else 'cuda')
        model.conf = 0.25
        model.iou = 0.45
        print(f"Model loaded successfully from {PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)


    results = model(Image.open(io.BytesIO(file)))

    pathlib.PosixPath = temp

    json_results = results_to_json(results, model)

    return json_results


def results_to_json(results, model):
    return [
        [
          {
          "class": int(pred[5]),
          "class_name": model.model.names[int(pred[5])],
          "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
          "confidence": float(pred[4]),
          }
        for pred in result
        ]
      for result in results.xyxy
      ]

# @app.get('/check-model')
# async def check_model():
#     try:
#         PATH = './model/best.pt'
#         # Load model
#         model = torch.hub.load('ultralytics/yolov5', 'custom', 
#                              path=PATH, 
#                              force_reload=True,
#                              device='cpu' if not torch.cuda.is_available() else 'cuda')
        
#         # Get model info
#         model_info = {
#             "model_path": PATH,
#             "exists": pathlib.Path(PATH).exists(),
#             "file_size": pathlib.Path(PATH).stat().st_size if pathlib.Path(PATH).exists() else 0,
#             "classes": model.names if hasattr(model, 'names') else [],
#             "device": str(next(model.parameters()).device),
#             "conf_threshold": model.conf,
#             "iou_threshold": model.iou
#         }
        
#         # Test with a blank image
#         test_img = Image.new('RGB', (640, 640))
#         test_results = model(test_img)
        
#         return {
#             "status": "success",
#             "model_info": model_info,
#             "can_inference": True,
#             "message": "Model loaded and tested successfully"
#         }
        
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "message": "Model verification failed"
#         }


    

