from typing import List
from fastapi import FastAPI, File, Depends, HTTPException, UploadFile, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sys import platform
from PIL import Image
from contextlib import asynccontextmanager
import torch
import io
import os
from openai import OpenAI
from datetime import datetime
from fastapi.responses import FileResponse

from sqlalchemy.orm import Session
from sqlalchemy import desc
from database import Base, engine, get_db
from pathlib import Path

from app_models import Diagnosis, Patient, ScanImage, NoduleObject
from schemas import ScanImageCreate, NoduleCreate, DiagnosisCreate, BoundingBox, BoundingBoxRequest
import json
import boto3
import uuid
from datetime import datetime


# Create all tables
Base.metadata.create_all(bind=engine)   

# Directory for temporary local file storage
LOCAL_UPLOAD_DIR = Path("uploads")
LOCAL_UPLOAD_DIR.mkdir(exist_ok=True)

#Uncomment if running on Windows
if platform == "win32":
    import pathlib


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

# Open AI Connection
client = OpenAI(
    api_key=config["OPENAI_API_KEY"],
)

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
def view_diagnosis(db: Session = Depends(get_db)):
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
            "date": diagnosis.diagnosis_date,
            "photo_path": image.photo_path,
            "nodules": nodule.properties.get("nodules"),
            "doctor_note": [nodule.properties.get("nodules")[i].get("doctor_note") for i in range(len(nodule.properties.get("nodules")))] 
                if len(nodule.properties.get("nodules")) > 0 else None , #   Return all notes from each nodule, if there are any
            }
        for diagnosis, patient, image, nodule in results
    ]

    return response

# API endpoint to get diagnosis by id
@app.get('/view-diagnosis/{diagnosis_id}')
def view_diagnosis(diagnosis_id: int, db: Session = Depends(get_db)):
    # Query the database
    result = db.query(Diagnosis, Patient, ScanImage, NoduleObject)\
        .join(ScanImage, Diagnosis.image_id == ScanImage.image_id)\
        .join(Patient, ScanImage.patient_id == Patient.patient_id)\
        .join(NoduleObject, Diagnosis.nodule_id == NoduleObject.nodule_id)\
        .filter(Diagnosis.diagnosis_id == diagnosis_id)\
        .first()
    
    diagnosis,patient,image,nodule = result

    # Format the response
    response = {
        "diagnosis_id": diagnosis.diagnosis_id,
        "patient_id": patient.patient_id,
        "patient_first_name": patient.firstname,
        "patient_last_name": patient.lastname,
        "date": diagnosis.diagnosis_date,
        "photo_path": image.photo_path,
        "nodules": nodule.properties.get("nodules")
    }

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
# @app.post('/upload')
# async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):

#     global model
#     if model is None:
#         return {"error":"Model not initialized"}
    
#     contents = await file.read()
    
#     image = Image.open(io.BytesIO(contents))

#     # Save the image to local storage
#     unique_image_name = f"{uuid.uuid4()}.{image.format.lower()}"
#     local_file_path = LOCAL_UPLOAD_DIR / unique_image_name
#     image.save(local_file_path)

#     results = model(image)

#     if platform == "win32":
#         pathlib.PosixPath = temp         

#     json_results = results_to_json(results, model)

#     # adding new image entry
#     new_user_id = 1 # placeholder, gonna be updated later after login function is implemented
#     new_patient_id = 1 # placeholder, gonna be updated later after clearance on how patient info is obtained
#     new_image_type_id = 1 # placeholder, gonna be updated later after clearance on how to store images, currently set to "RAW"
#     new_image_path = local_file_path
#     new_image_name = unique_image_name
#     new_upload_date = datetime.today().strftime('%Y-%m-%d')
#     new_description = "" # placeholder, gonna be updated later after clearance on how to add image descriptions, currently set to an empty string
#     new_image_format = image.format.lower()
    
#     # Prepping the metadata of the uploaded image
#     image_to_push = ScanImageCreate(user_id=new_user_id,
#                                     patient_id=new_patient_id,
#                                     image_type_id=new_image_type_id,
#                                     image_name=new_image_name,
#                                     description=new_description,
#                                     upload_date=new_upload_date,
#                                     file_format=new_image_format,
#                                     photo_path=str(new_image_path))
    
#     # Push the data onto the database
#     await push_image(image_to_push, db)

#     # Get the latest image id that was recently added
#     latest_image_id = await get_latest_image_id(db)

#     # Create list of nodules
#     nodules = []

#     # Iterate through each nodule object returned by the AI after analyzing the uploaded image and add those nodules metadata into the db
#     for index in range(len(json_results[0])):
#         # Prepping the information needed for the nodule
       

#         # # Creating the nodule to add into the db
#         # nodule_to_add = NoduleCreate(image_id=latest_image_id,
#         #                              nodule_type_id=new_nodule_type_id,
#         #                              position=new_position,
#         #                              doctor_note=new_doctor_note,
#         #                              intensity=new_intensity,
#         #                              size=new_size)
        
#         # # add the ongoing specified nodule
#         # await push_nodule(nodule_to_add, db)

#         print(json_results[0][index].get('bbox'))
#         print(json_results[0][index].get('confidence'))

#         nodules.append({"position": json_results[0][index].get('bbox'),
#                         "confidence": json_results[0][index].get('confidence')})
    
#     print(nodules)
#     new_nodule_type_id = 1 # Placeholder
#     # new_position = json_results[0][index].get('bbox')
#     new_doctor_note = "Lung Problem" # Placeholder
#     new_intensity = "Moderate" # Placeholder
#     new_size = "Small" # Placeholder


#     # Creating the nodule to add into the db
#     nodule_to_add = NoduleCreate(image_id=latest_image_id,
#                                      nodule_type_id=new_nodule_type_id,
#                                      position=[],
#                                      doctor_note=new_doctor_note,
#                                      intensity=new_intensity,
#                                      size=new_size,
#                                      properties={"nodules": nodules})
    
#     # add the ongoing specified nodule
#     await push_nodule(nodule_to_add, db)
        
#     # Prepping the information needed for diagnosis
#     latest_nodule_id = await get_latest_nodule_id(db)
#     new_status = "" # Placeholder
#     new_diagnosis_description = "" # Placeholder 

#     try:
#         # Upload the image to S3
#         s3_key = f"uploads/{uuid.uuid4()}.{image.format.lower()}"
#         with local_file_path.open("rb") as f:
#             s3.upload_fileobj(f, BUCKET_NAME, s3_key)

        
        
#         # Creating the diagnosis to add
#         diagnosis_to_add = DiagnosisCreate(image_id=latest_image_id,
#                                            nodule_id=latest_nodule_id,
#                                            status=new_status,
#                                            diagnosis_description=new_diagnosis_description) 
        
#         await push_diagnosis(diagnosis_to_add, db)

#         return nodules
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Add get_latest_diagnosis_id helper function

async def get_latest_diagnosis_id(db: Session = Depends(get_db)):
    try:
        latest_diagnosis_id = db.query(Diagnosis.diagnosis_id).order_by(desc(Diagnosis.diagnosis_id)).first()
        if latest_diagnosis_id:
            return latest_diagnosis_id[0]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post('/upload')
async def upload(patient_id: str, files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    global model
    if model is None:
        return {"error": "Model not initialized"}
    
    all_results = []
    
    for f in files:
        try:
            content = await f.read()
            image = Image.open(io.BytesIO(content))

            # Save the image to local storage under patient id's folder
            unique_image_name = f"{uuid.uuid4()}.{image.format.lower()}"
            patient_dir = LOCAL_UPLOAD_DIR / patient_id
            patient_dir.mkdir(exist_ok=True)
            local_file_path = patient_dir / unique_image_name
            image.save(local_file_path)

            # Process image with AI model
            results = model(image)
            if platform == "win32":
                pathlib.PosixPath = temp         
            json_results = results_to_json(results, model)

            # Create image entry
            image_to_push = ScanImageCreate(
                user_id=1,  # placeholder
                patient_id=patient_id,
                image_type_id=1,  # placeholder
                image_name=unique_image_name,
                description="",
                upload_date=datetime.today().strftime('%Y-%m-%d'),
                file_format=image.format.lower(),
                photo_path=str(local_file_path)
            )
            
            await push_image(image_to_push, db)
            latest_image_id = await get_latest_image_id(db)
            latest_diagnosis_id = await get_latest_diagnosis_id(db)

            # Process nodules for this image
            nodules = []
            for detection in json_results[0]:  # Access first batch's detections
                nodules.append({
                    "position": detection['bbox'],
                    "confidence": detection['confidence']
                })


            # Use GPT to generate a brief diagnosis
            diagnosis = generate_diagnosis(f"{patient_id}/{unique_image_name}", nodules)
            nodules_data = diagnosis["nodules"]  # Extract nodules data from GPT response
            
            for i, nodule in enumerate(nodules):
                if i < len(nodules_data):
                    gpt_data = nodules_data[i]  # Get the corresponding nodule from GPT response
                    nodule["severity"] = gpt_data["malignancy_risk"]
                    nodule["doctor_note"] = gpt_data["justification"] + " " + gpt_data["recommendation"]



            
            # Create nodule entry
            nodule_to_add = NoduleCreate(
                image_id=latest_image_id,
                nodule_type_id=1,  # Placeholder
                position=[],
                doctor_note="",  # Placeholder
                intensity="",  # Placeholder
                size="",  # Placeholder
                properties={"nodules": nodules}
            )
            
            await push_nodule(nodule_to_add, db)
            latest_nodule_id = await get_latest_nodule_id(db)

            # Upload to S3
            s3_key = f"uploads/{patient_id}/{unique_image_name}"
            with local_file_path.open("rb") as f:
                s3.upload_fileobj(f, BUCKET_NAME, s3_key)

            # Create diagnosis entry
            diagnosis_to_add = DiagnosisCreate(
                image_id=latest_image_id,
                nodule_id=latest_nodule_id,
                status="",  # Placeholder
                diagnosis_date=datetime.today().strftime('%Y-%m-%d'),
                diagnosis_description=""  # Placeholder
            )
            
            await push_diagnosis(diagnosis_to_add, db)

            all_results.append({
                "diagnosis_id": latest_diagnosis_id,
                "image_name": unique_image_name,
                "nodules": nodules
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {f.filename}: {str(e)}")

    return all_results


@app.put('/update-nodules')
async def update_nodules(bounding_box_request: BoundingBoxRequest, db: Session = Depends(get_db)):
    try:
        # Retrieve the current diagnosis_id
        # Create a new NoduleObject based on the bounding box
        # Push that NoduleObject onto the database

        diagnosis_id = bounding_box_request.diagnosis_id
        new_nodule_list = bounding_box_request.bounding_box_list

        current_nodules_props = await get_current_nodules_props(diagnosis_id, db)
        updated_nodules_props = [] #Empty so that we can append the new nodule properties (allow delete and update of nodules)

        for entries in new_nodule_list:
            # Process the information of the bounding box
            new_nodule_position = [entries.xCenter, entries.yCenter, entries.width, entries.height]
            new_properties = {
                'position': new_nodule_position,
                'confidence': entries.confidence,
                'severity': entries.severity,
                'doctor_note': entries.doctor_note,
            }
            
            updated_nodules_props.append(new_properties)
            
        
        current_nodules_props['nodules'] = updated_nodules_props
        print("--------------------------")
        print(updated_nodules_props)
        print(current_nodules_props)
        print("--------------------------")

        # if not current_nodules:
        #     raise HTTPException(status_code=404, detail="Nodule not found")

        # # Initialize properties array if it's null
        # if current_nodules is None:
        #     current_nodules = []

        # # current_nodules.append(new_nodules)
        status = await update_nodules_props(diagnosis_id, current_nodules_props, db)

        return status




    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def get_current_nodules_props(diagnosis_id: int, db: Session):
    try:
        nodule_id = db.query(Diagnosis.nodule_id).filter(Diagnosis.diagnosis_id == diagnosis_id).first()
        current_nodules_props = db.query(NoduleObject.properties).filter(NoduleObject.nodule_id == nodule_id[0]).first()

        return current_nodules_props[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


async def update_nodules_props(diagnosis_id: int, props: list, db: Session):
    try:
        # Find a way to optimize this, we don't want it to be query twice in one run of the request
        nodule_id = db.query(Diagnosis.nodule_id).filter(Diagnosis.diagnosis_id == diagnosis_id).first() 

        # UPDATE table_name SET column_name = new_value WHERE condition;
        nodule_obj = db.query(NoduleObject).filter(NoduleObject.nodule_id == nodule_id[0]).first()
        print("======================")
        print(nodule_obj)
        print("======================")

        nodule_obj.properties = props
        db.commit()

        return {
            "status": "Successfully updated the database."
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occured: {str(e)}")

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
    print("Pushing nodule")
    try:
        new_nodule = NoduleObject(image_id=nodule.image_id,
                                  nodule_type_id=nodule.nodule_type_id,
                                  doctor_note=nodule.doctor_note,
                                  intensity=nodule.intensity,
                                  size=nodule.size,
                                  properties=nodule.properties)

        db.add(new_nodule)
        db.commit()
        db.refresh(new_nodule)

        return {
            "message": "Nodule added successfully.",
            "data": {
                "image_id": new_nodule.image_id,
                "nodule_type_id": new_nodule.nodule_type_id,
                "properties": new_nodule.properties,
                "doctor_note": new_nodule.doctor_note,
                "intensity": new_nodule.intensity,
                "size": new_nodule.size,
            }
        }
    
    except Exception as e:
        print(e)
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
                                  diagnosis_date=datetime.today().strftime('%Y-%m-%d'),
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
                "diagnosis_description": new_diagnosis.diagnosis_description,
                "diagnosis_date": new_diagnosis.diagnosis_date
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
"""
Get uploaded Image from local storage
"""
app.mount("/uploads", StaticFiles(directory=LOCAL_UPLOAD_DIR), name="uploads")

@app.get('/images/{patient_id}/{image_name}')
async def get_image(patient_id: str, image_name: str, db: Session = Depends(get_db)):
    image = db.query(ScanImage).filter(
        ScanImage.image_name == image_name,
        ScanImage.patient_id == patient_id
    ).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Construct path including patient_id directory
    relative_path = LOCAL_UPLOAD_DIR/ patient_id / image_name
    print(relative_path)
    # if not relative_path.is_file():
    #     return {"error": "Image not found on the server"}
    
    return FileResponse(relative_path)


"""
    Diagnosis pagination
"""

def get_pagination_params(
        page: int = Query(1,gt=0),

        per_page: int = Query(10, gt=0),
):
    return {"page": page, "per_page": per_page}

@app.get('/diagnosis')
def get_diagnosis(
        pagination: dict = Depends(get_pagination_params),
        db: Session = Depends(get_db)
):
    page = pagination.get("page")
    per_page = pagination.get("per_page")

    # Calculate offset
    offset = (page - 1) * per_page

    results = db.query(Diagnosis, Patient, ScanImage, NoduleObject)\
        .join(ScanImage, Diagnosis.image_id == ScanImage.image_id)\
        .join(Patient, ScanImage.patient_id == Patient.patient_id)\
        .join(NoduleObject, Diagnosis.nodule_id == NoduleObject.nodule_id)\
        .offset(offset).limit(per_page).all()
        

    # Format the response
    response = [
            {
            "diagnosis_id": diagnosis.diagnosis_id,
            "patient_first_name": patient.firstname,
            "patient_last_name": patient.lastname,
            "date": diagnosis.diagnosis_date,
            "photo_path": image.photo_path,
            "nodules": nodule.properties.get("nodules"),
            "doctor_note": [nodule.properties.get("nodules")[i].get("doctor_note") for i in range(len(nodule.properties.get("nodules")))] 
                if len(nodule.properties.get("nodules")) > 0 else None , #   Return all notes from each nodule, if there are any
            }
        for diagnosis, patient, image, nodule in results
    ]

    return response

"""
Use GPT to generate a diagnosis based on the nodule properties.
"""
def generate_diagnosis(image_url,detections):
    print("Generating diagnosis...")
    print(f"https://caf3-116-111-184-66.ngrok-free.app/images/{image_url}")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI radiology assistant."},
            {"role": "user", "content": f""" 
            Ảnh CT phổi sau đã được khoanh vùng bằng mô hình AI.
            Vui lòng phân tích vị trí nốt phổi trong ảnh và kết hợp với dữ liệu bệnh nhân (nếu có):
            - Tuổi: {"Không rõ"}
            - Giới tính: {"Không rõ"}
            - Tiền sử hút thuốc: {"Không rõ"}
            - Triệu chứng: {"Không rõ"}
            - Có ảnh CT cũ để so sánh: {"Không rõ"}

            Hãy đánh giá nguy cơ ác tính của từng nốt phổi và trả về kết quả theo định dạng JSON với các thông tin sau:
            - `nodules`: danh sách các nốt với thông tin:
            - `position`: vị trí nốt trên ảnh
            - `malignancy_risk`: nguy cơ ác tính BẰNG TIẾNG ANH (mild, moderate, severe, critical)
            - `justification`: lý do đánh giá nguy cơ BẰNG TIẾNG VIỆT (giải thích chi tiết về đặc điểm hình ảnh, vị trí, kích thước và các yếu tố nguy cơ)
            - `recommendation`: hướng xử lý tiếp theo BẰNG TIẾNG VIỆT (đề xuất cụ thể về theo dõi, sinh thiết hoặc phẫu thuật dựa theo hướng dẫn y khoa)

            Trả lời **CHỈ** dưới dạng JSON thuần túy, không thêm markdown, code blocks hoặc backticks.
            
            Nốt phổi: {detections}""",
         "image": {"type": "image_url", "image_url": f"https://caf3-116-111-184-66.ngrok-free.app/images/{image_url}"}},
        ],
        max_tokens=1500,
        temperature=0.3
    )

    result = response.choices[0].message.content
    print(f"Raw GPT response: {result}")

    if result.startswith("```"):
            # Find the position of the first newline after the opening ```
            first_newline = result.find("\n")
            if first_newline != -1:
                # Find the position of the closing ```
                closing_backticks = result.rfind("```")
                if closing_backticks > first_newline:
                    # Extract content between the backticks, ignoring the first line (```json)
                    result = result[first_newline+1:closing_backticks].strip()
                else:
                    # If closing backticks not found, take everything after first line
                    result = result[first_newline+1:].strip()
        

    try:
        parsed_json = json.loads(result)
        print(f"Parsed JSON: {parsed_json}")
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {
            "error": "Failed to decode JSON response from GPT."
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "error": "An unexpected error occurred."
        }
    
    
