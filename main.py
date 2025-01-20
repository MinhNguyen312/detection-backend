from fastapi import FastAPI, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from contextlib import asynccontextmanager
import torch
import io
import pathlib

from sqlalchemy.orm import Session
from sqlalchemy import desc
from database import Base, engine, get_db
from pathlib import Path

from app_models import Diagnosis, Patient, ScanImage, NoduleObject
from schemas import ScanImageCreate, NoduleCreate, DiagnosisCreate
import json
import boto3
import uuid
from datetime import datetime


# Create all tables
Base.metadata.create_all(bind=engine)   

# Directory for temporary local file storage
LOCAL_UPLOAD_DIR = Path("uploads")
LOCAL_UPLOAD_DIR.mkdir(exist_ok=True)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# On startup, load all config for S3 bucket
def load_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

config = load_config("config.json")

s3 = boto3.client(
    "s3",
    aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)

# S3 bucket name
BUCKET_NAME = "lung.images"


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

"""
Obtain a list of diagnoses with the patient's name, date of diagnosis, path of the patient's scanned image and the position of the nodules.

Returns:
    A json in this format:
        [
            {
                "diagnosis_id": ,
                "patient_first_name": "",
                "patient_last_name": "",
                "date": "",
                "photo_path": "",
                "nodule_position": []
            }
        ]

Example:
    When the GET request is sent, the API returns:
        [
            {
                "diagnosis_id": 1,
                "patient_first_name": "John",
                "patient_last_name": "Doe",
                "date": "15/10/2024",
                "photo_path": "/path/to/images/P001_RAW.jpg",
                "nodule_position": [
                    0,
                    0.6225961538461539,
                    0.7127403846153846,
                    0.08774038461538461,
                    0.07451923076923077
                ]
            }
        ]
"""
@app.get('/view-diagnosis')
def view_diagonsis(db: Session = Depends(get_db)):
    # Query the database
    results = db.query(Diagnosis, Patient, ScanImage, NoduleObject)\
        .join(ScanImage, Diagnosis.image_id == ScanImage.image_id)\
        .join(Patient, ScanImage.patient_id == Patient.patient_id)\
        .join(NoduleObject, Diagnosis.nodule_id == NoduleObject.nodule_id)\
        .all()

    # Format the response
    response = [
        {
            "diagnosis_id": diagnosis.diagnosis_id,
            "patient_first_name": patient.firstname,
            "patient_last_name": patient.lastname,
            "date": "15/10/2024", # placeholder only
            "photo_path": image.photo_path,
            "nodule_position": nodule.position,
        }
        for diagnosis, patient, image, nodule in results
    ]

    return response


PATH = './model/best.pt'
model = None


"""
Takes in an upload image from the user, feed it into the AI to retrieve coordinates of the nodules identified by the AI. After that process is
done, move on by saving the upload image into the local server, as well as on the Cloud with S3. Meanwhile, the image metadata and relevant
information such as diagosis and nodule objects are also recorded and put into the database.

Args:
    file (bytes): the uploaded file
    db (Session): using the generated db with the pre-defined credentials

Returns:
    The prediction of the AI in JSON format.

Example: 
    When an image is uploaded:
        [
            [
                {
                "class": 0,
                "class_name": "Nodules",
                "bbox": [
                    145,
                    279,
                    161,
                    295
                ],
                "confidence": 0.5544038414955139
                },
                {
                "class": 0,
                "class_name": "Nodules",
                "bbox": [
                    116,
                    283,
                    128,
                    295
                ],
                "confidence": 0.5108078122138977
                }
            ]
        ]
"""
@app.post('/upload')
async def upload(file: bytes = File(...), db: Session = Depends(get_db)):

    global model
    if model is None:
        return {"error":"Model not initialized"}
    
    image = Image.open(io.BytesIO(file))

    # Save the image to local storage
    unique_image_name = f"{uuid.uuid4()}.{image.format.lower()}"
    local_file_path = LOCAL_UPLOAD_DIR / unique_image_name
    image.save(local_file_path)

    results = model(image)

    pathlib.PosixPath = temp

    json_results = results_to_json(results, model)

    # adding new image entry
    new_user_id = 1 # placeholder, gonna be updated later after login function is implemented
    new_patient_id = 1 # placeholder, gonna be updated later after clearance on how patient info is obtained
    new_image_type_id = 1 # placeholder, gonna be updated later after clearance on how to store images, currently set to "RAW"
    new_image_path = local_file_path
    new_image_name = unique_image_name
    new_upload_date = datetime.today().strftime('%Y-%m-%d')
    new_description = "" # placeholder, gonna be updated later after clearance on how to add image descriptions, currently set to an empty string
    new_image_format = image.format.lower()
    
    # Prepping the metadata of the uploaded image
    image_to_push = ScanImageCreate(user_id=new_user_id,
                                    patient_id=new_patient_id,
                                    image_type_id=new_image_type_id,
                                    image_name=new_image_name,
                                    description=new_description,
                                    upload_date=new_upload_date,
                                    file_format=new_image_format,
                                    photo_path=str(new_image_path))
    
    # Push the data onto the database
    await push_image(image_to_push, db)

    # Iterate through each nodule object returned by the AI after analyzing the uploaded image and add those nodules metadata into the db
    for index in range(len(json_results[0])):
        # Get the latest image id that was recently added
        latest_image_id = await get_latest_image_id(db)

        # Prepping the information needed for the nodule
        new_nodule_type_id = 1 # Placeholder
        new_position = json_results[0][index].get('bbox')
        new_doctor_note = "Lung Problem" # Placeholder
        new_intensity = "Moderate" # Placeholder
        new_size = "Small" # Placeholder

        # Creating the nodule to add into the db
        nodule_to_add = NoduleCreate(image_id=latest_image_id,
                                     nodule_type_id=new_nodule_type_id,
                                     position=new_position,
                                     doctor_note=new_doctor_note,
                                     intensity=new_intensity,
                                     size=new_size)
        
        # add the ongoing specified nodule
        await push_nodule(nodule_to_add, db)
        
        # Prepping the information needed for diagnosis
        latest_nodule_id = await get_latest_nodule_id(db)
        new_status = "" # Placeholder
        new_diagnosis_description = "" # Placeholder 

        # Creating the diagnosis to add
        diagnosis_to_add = DiagnosisCreate(image_id=latest_image_id,
                                           nodule_id=latest_nodule_id,
                                           status=new_status,
                                           diagnosis_description=new_diagnosis_description) 
        
        await push_diagnosis(diagnosis_to_add, db)

    try:
        # Upload the image to S3
        s3_key = f"uploads/{uuid.uuid4()}.{image.format.lower()}"
        with local_file_path.open("rb") as f:
            s3.upload_fileobj(f, BUCKET_NAME, s3_key)

        return json_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


    
"""
Format the predicted data returned by the AI into a JSON format.

Args:
    results (List[List[Dict[str, any]]]): the result returned by the AI
    model (object): the AI model

Returns:
    The passed in result in JSON format
"""
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

"""
Push the passed in image's metadata onto the database.

Args:
    image (ScanImageCreate): the DTO that carries the image's metadata
    db (Session): using the generated db with the pre-defined credentials

Returns:
    A JSON message with the passed in metadata.
"""
async def push_image(image: ScanImageCreate, db: Session):
    try:
        new_image = ScanImage(user_id = image.user_id, 
                            patient_id = image.patient_id,
                            image_type_id = image.image_type_id,
                            image_name = image.image_name,
                            description = image.description,
                            upload_date = image.upload_date,
                            file_format = image.file_format,
                            photo_path = image.photo_path)
        
        # Add the new record to the session and commit it
        db.add(new_image)
        db.commit()
        db.refresh(new_image) # Refresh the instance to retrieve the generated ID
    
        # Return the newly created record
        return {
            "message": "Scan image added successfully.",
            "data": {
                "image_id": new_image.image_id,
                "user_id": new_image.user_id,
                "patient_id": new_image.patient_id,
                "image_type_id": new_image.image_type_id,
                "description": new_image.description,
                "upload_date": new_image.upload_date,
                "file_format": new_image.file_format,
                "photo_path": new_image.photo_path
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

"""
Returns the id of the latest image that was put into the database.

Args:
    db (Session): using the generated db with the pre-defined credentials

Returns:
    The id of the latest image (int) or None if failed to query.
"""
async def get_latest_image_id(db: Session):
    try: 
        latest_image_id = db.query(ScanImage.image_id).order_by(desc(ScanImage.image_id)).first()

        # Extract the integer value from the tuple
        if latest_image_id:
            return latest_image_id[0]  # Get the first (and only) value from the tuple
        else:
            return None  # Handle the case where no rows are returned

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occured: {str(e)}" )

async def push_nodule(nodule: NoduleCreate, db: Session):
    try:
        new_nodule = NoduleObject(image_id=nodule.image_id,
                                  nodule_type_id=nodule.nodule_type_id,
                                  position=nodule.position,
                                  doctor_note=nodule.doctor_note,
                                  intensity=nodule.intensity,
                                  size=nodule.size)
        
        db.add(new_nodule)
        db.commit()
        db.refresh(new_nodule)

        return {
            "message": "Nodule added successfully.",
            "data": {
                "image_id": new_nodule.image_id,
                "nodule_type_id": new_nodule.nodule_type_id,
                "position": new_nodule.position,
                "doctor_note": new_nodule.doctor_note,
                "intensity": new_nodule.intensity,
                "size": new_nodule.size,


            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

"""
Returns the id of the latest nodule object that is put into the database.

Args:
    db (Session): using the generated db with the pre-defined credentials

Returns:
    The id of the latest nodule object (int) or None if failed to query.
"""
async def get_latest_nodule_id(db: Session):
    try:
        latest_nodule_id = db.query(NoduleObject.nodule_id).order_by(desc(NoduleObject.nodule_id)).first()

        # Extract the integer value from the tuple
        if latest_nodule_id:
            return latest_nodule_id[0]  # Get the first (and only) value from the tuple
        else:
            return None  # Handle the case where no rows are returned

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occured: {str(e)}" )


"""
Insert a diagnosis record into the database.

Args:
    diagnosis (DiagnosisCreate): the DTO for the diagnosis' metadata.

Returns:
    A JSON message with the added diagnosis' metadata.
"""
async def push_diagnosis(diagnosis: DiagnosisCreate, db: Session):
    try:
        new_diagnosis = Diagnosis(image_id=diagnosis.image_id,
                                  nodule_id=diagnosis.nodule_id,
                                  status=diagnosis.status,
                                  diagnosis_description=diagnosis.diagnosis_description)
        
        db.add(new_diagnosis)
        db.commit()
        db.refresh(new_diagnosis)

        return {
            "message": "Diagnosis added successfully.",
            "data": {
                "image_id": new_diagnosis.image_id,
                "nodule_id": new_diagnosis.nodule_id,
                "status": new_diagnosis.status,
                "diagnosis_description": new_diagnosis.diagnosis_description


            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")