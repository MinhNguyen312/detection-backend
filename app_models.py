from sqlalchemy import Column, Integer, String, Float, Date, JSON
from database import Base

class Diagnosis(Base):
    __tablename__ = "diagnosis"
    __table_args__ = {"schema": "lung_testing"}  # Specify schema separately
    diagnosis_id = Column(Integer, primary_key=True, index=False)
    image_id = Column(Integer)
    nodule_id = Column(Integer)
    status = Column(String)
    diagnosis_description = Column(String)

class Patient(Base):
    __tablename__ = "patient"
    __table_args__ = {"schema": "lung_testing"}  # Specify schema separately
    patient_id = Column(Integer, primary_key=True, index=False)
    firstname = Column(String)
    lastname = Column(String)
    dob = Column(Date)
    sex = Column(String)
    phone = Column(String)
    email = Column(String)
    address = Column(String)
    health_insurance = Column(String)
    personal_identity = Column(String)

class ScanImage(Base):
    __tablename__ = "image"
    __table_args__ = {"schema": "lung_testing"}  # Specify schema separately
    image_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer)
    patient_id = Column(Integer)
    image_type_id = Column(Integer)
    image_name = Column(String)
    description = Column(String)
    upload_date = Column(Date)
    file_format = Column(String)
    photo_path = Column(String)

    def __repr__(self):
        return (
            f"<ScanImage(image_id={self.image_id}, user_id={self.user_id}, "
            f"patient_id={self.patient_id}, image_type_id={self.image_type_id}, "
            f"description={self.description}, upload_date={self.upload_date}, "
            f"file_format={self.file_format}, photo_path={self.photo_path})>"
        )

class NoduleObject(Base):
    __tablename__ = "noduleobject"
    __table_args__ = {"schema": "lung_testing"}  # Specify schema separately
    nodule_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    image_id = Column(Integer)
    nodule_type_id = Column(Integer)
    position = Column(Float)
    doctor_note = Column(String)
    intensity = Column(String)
    size = Column(String)
    parameters = Column(JSON)


