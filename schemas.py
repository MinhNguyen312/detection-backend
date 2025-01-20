from pydantic import BaseModel
from datetime import date
from typing import List

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
    position: List[float]
    doctor_note: str
    intensity: str
    size: str

class DiagnosisCreate(BaseModel):
    image_id: int
    nodule_id: int
    status: str
    diagnosis_description: str
