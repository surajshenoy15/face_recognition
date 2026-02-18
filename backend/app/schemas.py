from pydantic import BaseModel

class EnrollResponse(BaseModel):
    success: bool
    student_id: str
    meta: dict

class VerifyResponse(BaseModel):
    success: bool
    student_id: str
    match: bool
    similarity: float
    threshold: float
    meta: dict

class IdentifyResponse(BaseModel):
    success: bool
    identified: bool
    student_id: str | None
    best_similarity: float
    threshold: float
    meta: dict
