from pydantic import BaseModel
from datetime import date
from typing import List, Dict, Any

class ScanImageCreate(BaseModel):
    user_id: int
    patient_id: int
    image_type_id: int
    image_name: str
    description: str
    upload_date: date
    file_format: str
    photo_path: str

class NoduleCreate(BaseModel):
    image_id: int
    nodule_type_id: int
    properties: List[Dict[str, Any]]
    doctor_note: str
    intensity: str
    size: str

class DiagnosisCreate(BaseModel):
    image_id: int
    nodule_id: int
    status: str
    diagnosis_description: str

class BoundingBox(BaseModel):
    classId: int
    className: str
    # bbox: List[int]
    confidence: int
    xCenter: float
    yCenter: float
    width: float
    height: float
    isTemp: bool # no need to store this into db
    severity: str
    notes: str

class BoundingBoxRequest(BaseModel):
    image_id: int
    bounding_box_list: List[BoundingBox]