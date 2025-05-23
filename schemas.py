from pydantic import BaseModel
from datetime import date
from typing import List, Dict, Any, Optional

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
    properties: Optional[Dict[str, Any]] = None  # New field for JSONB

class DiagnosisCreate(BaseModel):
    image_id: int
    nodule_id: int
    status: str
    diagnosis_description: str
    diagnosis_date: date

class BoundingBox(BaseModel):
    classId: int
    className: str
    # bbox: List[int]
    confidence: float
    xCenter: float
    yCenter: float
    width: float
    height: float
    isTemp: bool # no need to store this into db
    severity: str
    doctor_note: str

class BoundingBoxRequest(BaseModel):
    diagnosis_id: int
    bounding_box_list: List[BoundingBox]