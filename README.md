# face_recognition# Face Recognition API

Student face enrollment and activity verification using **YuNet + SFace** (Apache 2.0 — ✅ Commercial use allowed).

---

## Project Structure

```
face_recognition/
├── main.py
├── requirements.txt
├── face_recognition.db        ← auto-created on first run
└── app/
    ├── __init__.py
    ├── database.py
    ├── models/
    │   ├── __init__.py
    │   ├── student.py
    │   └── face/
    │       ├── face_detection_yunet_2023mar.onnx     ← download manually
    │       └── face_recognition_sface_2021dec.onnx   ← download manually
    ├── routes/
    │   ├── __init__.py
    │   └── face_routes.py
    └── services/
        ├── __init__.py
        └── face_service.py
```

---

## Setup

### 1. Clone the repo
```powershell
git clone https://github.com/surajshenoy15/face_recognition.git
cd face_recognition
```

### 2. Create virtual environment and install dependencies
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download ONNX models
```powershell
# Create the folder
mkdir app\models\face

# Download YuNet (face detection)
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "app/models/face/face_detection_yunet_2023mar.onnx"

# Download SFace (face recognition)
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" -OutFile "app/models/face/face_recognition_sface_2021dec.onnx"
```

### 4. Run
```powershell
uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/docs** to test in browser.

---

## API Endpoints

### POST `/face/register`
Enroll a student face using 3–5 selfies from different angles.

**Form-data:**
| Key | Type | Value |
|-----|------|-------|
| student_id | Text | STU001 |
| images | File | 3–5 selfie images |

**Response:**
```json
{
  "success": true,
  "student_id": "STU001",
  "message": "Face registered using 3 image(s).",
  "photos_processed": 3,
  "photos_failed": 0
}
```

---

### POST `/face/verify`
Verify if a student is present in a group/activity photo.
Returns the image with a green bounding box on the matched face.

**Form-data:**
| Key | Type | Value |
|-----|------|-------|
| student_id | Text | STU001 |
| group_photo | File | group photo |

**Response:** Image with bounding box drawn on matched face.
Response headers contain:
- `X-Matched` — true/false
- `X-Cosine-Score` — similarity score
- `X-Total-Faces` — number of faces detected

---

### GET `/face/status/{student_id}`
Check if a student has registered their face.

---

### DELETE `/face/unregister/{student_id}`
Remove a student's face data from the database.

---

## Models

| Model | Task | Accuracy | License |
|-------|------|----------|---------|
| YuNet | Face Detection | 81.1% mAP | Apache 2.0 ✅ |
| SFace | Face Recognition | 96.7% LFW | Apache 2.0 ✅ |

---

## Tech Stack
- **FastAPI** — Backend
- **SQLite** — Database
- **OpenCV YuNet** — Face Detection
- **OpenCV SFace** — Face Recognition