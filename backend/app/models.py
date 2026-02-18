from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, func, Index
from app.db import Base

class StudentEmbedding(Base):
    __tablename__ = "student_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # float32[128] bytes

    created_at = Column(DateTime(timezone=True), server_default=func.now())

Index("ix_student_embeddings_student_id", StudentEmbedding.student_id)
