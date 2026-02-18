from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import json

from app.db import get_db
from app.student_model import StudentFaceEmbedding
from app.services.face_service import face_service

router = APIRouter(prefix="/face", tags=["Face Recognition"])


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class FaceRegistrationRequest(BaseModel):
    student_id: str = Field(..., example="STU001")
    email: EmailStr = Field(..., example="student@college.edu")
    images: list[str] = Field(
        ...,
        min_items=3,
        max_items=5,
        description="List of 3-5 base64-encoded selfie images (JPEG/PNG). "
                    "Capture from different angles: front, left, right, up, down.",
        example=["data:image/jpeg;base64,/9j/4AAQ..."]
    )

class FaceRegistrationResponse(BaseModel):
    success: bool
    student_id: str
    message: str
    photos_processed: int
    photos_failed: int

class ActivityVerificationRequest(BaseModel):
    student_id: str = Field(..., example="STU001")
    group_photo: str = Field(
        ...,
        description="Base64-encoded group photo where student should be present.",
        example="data:image/jpeg;base64,/9j/4AAQ..."
    )

class ActivityVerificationResponse(BaseModel):
    student_id: str
    matched: bool
    cosine_score: Optional[float]
    l2_score: Optional[float]
    total_faces_in_photo: int
    matched_face_box: Optional[list[int]]  # [x, y, w, h]
    message: str

class EmbeddingStatusResponse(BaseModel):
    student_id: str
    email: str
    registered: bool
    photo_count: Optional[int]
    registered_at: Optional[str]


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=FaceRegistrationResponse,
    summary="Register student face (3–5 selfies from different angles)",
)
def register_face(payload: FaceRegistrationRequest, db: Session = Depends(get_db)):
    """
    **Step 1: Student face enrollment.**
    
    - Send 3 to 5 selfie images in base64.
    - Capture from: front, left side, right side, slight up, slight down.
    - System extracts 128-d embeddings using SFace, averages them, and stores.
    - Calling this again for the same student_id **updates** their embedding.
    
    **Postman usage:**
    - Method: POST
    - Body: raw JSON
    ```json
    {
      "student_id": "STU001",
      "email": "student@college.edu",
      "images": ["data:image/jpeg;base64,...", "..."]
    }
    ```
    """
    embeddings = []
    failed = 0

    for i, img_b64 in enumerate(payload.images):
        try:
            result = face_service.extract_embedding(img_b64)
            embeddings.append(result["embedding"])
        except ValueError as e:
            failed += 1
            # Continue processing other images; don't fail entirely on one bad image

    if len(embeddings) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract face from enough images. "
                   f"Successfully processed: {len(embeddings)}, Failed: {failed}. "
                   f"Need at least 3 valid face images."
        )

    # Average all valid embeddings into one representative vector
    avg_embedding = face_service.average_embeddings(embeddings)

    # Upsert into DB
    existing = db.query(StudentFaceEmbedding).filter_by(student_id=payload.student_id).first()
    if existing:
        existing.set_embedding(avg_embedding)
        existing.email = payload.email
        existing.photo_count = len(embeddings)
        from datetime import datetime
        existing.updated_at = datetime.utcnow()
    else:
        record = StudentFaceEmbedding(
            student_id=payload.student_id,
            email=payload.email,
            photo_count=len(embeddings),
        )
        record.set_embedding(avg_embedding)
        db.add(record)

    db.commit()

    return FaceRegistrationResponse(
        success=True,
        student_id=payload.student_id,
        message=f"Face registered successfully using {len(embeddings)} images.",
        photos_processed=len(embeddings),
        photos_failed=failed,
    )


@router.post(
    "/verify-activity",
    response_model=ActivityVerificationResponse,
    summary="Verify student presence in group/activity photo",
)
def verify_activity(payload: ActivityVerificationRequest, db: Session = Depends(get_db)):
    """
    **Step 2: Verify student is in the activity group photo.**
    
    - Fetch stored embedding for student_id.
    - Detect all faces in the group photo.
    - Match each face against the stored embedding using cosine similarity.
    - Returns match result + which face box matched.
    
    **Postman usage:**
    - Method: POST
    - Body: raw JSON
    ```json
    {
      "student_id": "STU001",
      "group_photo": "data:image/jpeg;base64,..."
    }
    ```
    """
    # Fetch student embedding
    record = db.query(StudentFaceEmbedding).filter_by(student_id=payload.student_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No face registered for student_id '{payload.student_id}'. "
                   f"Student must complete face registration first."
        )

    stored_embedding = record.get_embedding()

    # Run matching
    result = face_service.match_face_in_group(payload.group_photo, stored_embedding)

    return ActivityVerificationResponse(
        student_id=payload.student_id,
        matched=result["matched"],
        cosine_score=result.get("cosine_score"),
        l2_score=result.get("l2_score"),
        total_faces_in_photo=result.get("total_faces", 0),
        matched_face_box=result.get("matched_face_box"),
        message=result.get("reason", ""),
    )


@router.get(
    "/status/{student_id}",
    response_model=EmbeddingStatusResponse,
    summary="Check if student has registered their face",
)
def get_face_status(student_id: str, db: Session = Depends(get_db)):
    """
    Check whether a student has completed face registration.
    Useful to gate activity submission until face is enrolled.
    """
    record = db.query(StudentFaceEmbedding).filter_by(student_id=student_id).first()
    if not record:
        return EmbeddingStatusResponse(
            student_id=student_id,
            email="",
            registered=False,
            photo_count=None,
            registered_at=None,
        )
    return EmbeddingStatusResponse(
        student_id=student_id,
        email=record.email,
        registered=True,
        photo_count=record.photo_count,
        registered_at=str(record.registered_at),
    )


@router.delete(
    "/unregister/{student_id}",
    summary="Remove stored face embedding for a student",
)
def unregister_face(student_id: str, db: Session = Depends(get_db)):
    """Delete stored face embedding (e.g., when student account is removed)."""
    record = db.query(StudentFaceEmbedding).filter_by(student_id=student_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="No face record found.")
    db.delete(record)
    db.commit()
    return {"success": True, "message": f"Face data removed for {student_id}"}