from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime
import base64
import cv2
import numpy as np
import io

from app.database import get_db
from app.models.student import StudentFace
from app.services import face_service

router = APIRouter(prefix="/face", tags=["Face Recognition"])


def file_to_b64(file_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(file_bytes).decode()


def draw_box(image_bytes: bytes, face_box: list, matched: bool) -> bytes:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if matched and face_box:
        x, y, w, h = face_box
        color = (0, 255, 0)
        label = "STUDENT"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), color, -1)
        cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif not matched:
        cv2.putText(img, "STUDENT NOT FOUND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@router.post("/register", summary="Enroll student face using 3-5 selfies")
async def register(
    student_id: str              = Form(..., example="STU001"),
    images:     list[UploadFile] = File(..., description="Upload 3 to 5 selfie images"),
    db: Session = Depends(get_db)
):
    if len(images) < 3 or len(images) > 5:
        raise HTTPException(status_code=422, detail="Send between 3 and 5 images.")

    embeddings, failed = [], 0

    for img_file in images:
        try:
            contents = await img_file.read()
            if not contents:
                failed += 1
                continue
            emb = face_service.extract_embedding(file_to_b64(contents))
            embeddings.append(emb)
        except ValueError:
            failed += 1

    if len(embeddings) < 3:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(embeddings)} usable face(s) found (need >= 3). "
                   f"{failed} image(s) had no detectable face."
        )

    avg_emb = face_service.average_embeddings(embeddings)

    record = db.query(StudentFace).filter_by(student_id=student_id).first()
    if record:
        record.set_embedding(avg_emb)
        record.photo_count = len(embeddings)
        record.updated_at  = datetime.utcnow()
    else:
        record = StudentFace(student_id=student_id, photo_count=len(embeddings))
        record.set_embedding(avg_emb)
        db.add(record)

    db.commit()
    return {
        "success":          True,
        "student_id":       student_id,
        "message":          f"Face registered using {len(embeddings)} image(s).",
        "photos_processed": len(embeddings),
        "photos_failed":    failed,
    }


@router.post("/verify", summary="Verify student in group/activity photo")
async def verify(
    student_id:  str        = Form(..., example="STU001"),
    group_photo: UploadFile = File(..., description="Group or activity photo"),
    db: Session = Depends(get_db)
):
    record = db.query(StudentFace).filter_by(student_id=student_id).first()
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No face registered for '{student_id}'. Register first."
        )

    contents = await group_photo.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = face_service.match_in_group(file_to_b64(contents), record.get_embedding())

    annotated_image = draw_box(
        contents,
        result.get("matched_face_box") if result["matched"] else None,
        result["matched"]
    )

    headers = {
        "X-Matched":      str(result["matched"]),
        "X-Cosine-Score": str(result.get("cosine_score", "")),
        "X-L2-Score":     str(result.get("l2_score", "")),
        "X-Total-Faces":  str(result.get("total_faces", 0)),
        "X-Message":      result.get("reason", ""),
    }

    return StreamingResponse(
        io.BytesIO(annotated_image),
        media_type="image/jpeg",
        headers=headers,
    )


@router.get("/status/{student_id}", summary="Check registration status")
def get_status(student_id: str, db: Session = Depends(get_db)):
    record = db.query(StudentFace).filter_by(student_id=student_id).first()
    if not record:
        return {"student_id": student_id, "registered": False}
    return {
        "student_id":    student_id,
        "registered":    True,
        "photo_count":   record.photo_count,
        "registered_at": str(record.registered_at),
    }


@router.delete("/unregister/{student_id}", summary="Remove student face data")
def unregister(student_id: str, db: Session = Depends(get_db)):
    record = db.query(StudentFace).filter_by(student_id=student_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="No record found.")
    db.delete(record)
    db.commit()
    return {"success": True, "message": f"Face data removed for {student_id}"}