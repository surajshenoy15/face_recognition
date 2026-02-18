"""
student_model.py - SQLAlchemy model for storing face embeddings
Place this in: backend/app/student_model.py
(Adds face_embedding column to your existing student setup)
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class StudentFaceEmbedding(Base):
    """
    Stores the averaged face embedding for each student.
    One row per student — updated when they re-register their face.
    """
    __tablename__ = "student_face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    embedding = Column(Text, nullable=False)          # JSON array of 128 floats
    photo_count = Column(Integer, default=0)          # How many selfies were used
    registered_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_embedding(self) -> list[float]:
        import json
        return json.loads(self.embedding)

    def set_embedding(self, emb: list[float]):
        import json
        self.embedding = json.dumps(emb)