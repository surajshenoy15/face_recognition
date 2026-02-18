from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from app.database import Base
import json


class StudentFace(Base):
    __tablename__ = "student_faces"

    id            = Column(Integer, primary_key=True, index=True)
    student_id    = Column(String(100), unique=True, index=True, nullable=False)
    embedding     = Column(Text, nullable=False)
    photo_count   = Column(Integer, default=0)
    registered_at = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_embedding(self) -> list:
        return json.loads(self.embedding)

    def set_embedding(self, emb: list):
        self.embedding = json.dumps(emb)