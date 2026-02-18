import cv2
import numpy as np
import base64
from pathlib import Path

# ── Model paths ──────────────────────────────────────────────────────────────
# Place your .onnx files inside: face_recognition/app/models/face/
MODEL_DIR        = Path(__file__).parent.parent / "models" / "face"
DETECTOR_MODEL   = str(MODEL_DIR / "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = str(MODEL_DIR / "face_recognition_sface_2021dec.onnx")

# ── Thresholds (from OpenCV SFace paper) ─────────────────────────────────────
COSINE_THRESHOLD = 0.35   # >= match
L2_THRESHOLD     = 1.1  # <= match


def _decode_image(image_b64: str) -> np.ndarray:
    """Base64 string → OpenCV BGR image."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Send a valid JPEG or PNG in base64.")
    return img


def _detect_faces(img: np.ndarray) -> np.ndarray | None:
    """Run YuNet face detector and return all face boxes."""
    h, w = img.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        DETECTOR_MODEL, "", (w, h),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )
    _, faces = detector.detect(img)
    return faces


def _get_recognizer():
    return cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")


def extract_embedding(image_b64: str) -> list[float]:
    """
    Detect the largest face in the image and return its 128-d SFace embedding.
    Raises ValueError if no face is found.
    """
    img   = _decode_image(image_b64)
    faces = _detect_faces(img)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected. Make sure your face is clearly visible.")

    # Pick largest face by area
    largest = faces[np.argmax(faces[:, 2] * faces[:, 3])]

    recognizer = _get_recognizer()
    aligned    = recognizer.alignCrop(img, largest)
    embedding  = recognizer.feature(aligned).flatten().tolist()
    return embedding


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Average N embeddings and L2-normalize the result."""
    arr  = np.array(embeddings, dtype=np.float32)
    avg  = arr.mean(axis=0)
    norm = np.linalg.norm(avg)
    return (avg / norm if norm > 0 else avg).tolist()


def match_in_group(group_image_b64: str, stored_embedding: list[float]) -> dict:
    """
    Find every face in a group photo and compare each against stored_embedding.
    Returns the best match result.
    """
    img   = _decode_image(group_image_b64)
    faces = _detect_faces(img)

    if faces is None or len(faces) == 0:
        return {
            "matched": False,
            "reason": "No faces detected in the group photo.",
            "cosine_score": None,
            "l2_score": None,
            "matched_face_box": None,
            "total_faces": 0,
        }

    recognizer  = _get_recognizer()
    stored      = np.array(stored_embedding, dtype=np.float32).reshape(1, -1)
    best_cosine = -1.0
    best_l2     = float("inf")
    best_box    = None

    for face in faces:
        try:
            aligned = recognizer.alignCrop(img, face)
            emb     = recognizer.feature(aligned)
            cosine  = recognizer.match(stored, emb, cv2.FaceRecognizerSF_FR_COSINE)
            l2      = recognizer.match(stored, emb, cv2.FaceRecognizerSF_FR_NORM_L2)
            if cosine > best_cosine:
                best_cosine = cosine
                best_l2     = l2
                best_box    = face[:4].astype(int).tolist()
        except Exception:
            continue

    matched = best_cosine >= COSINE_THRESHOLD and best_l2 <= L2_THRESHOLD

    return {
        "matched":          matched,
        "cosine_score":     round(float(best_cosine), 4),
        "l2_score":         round(float(best_l2), 4),
        "cosine_threshold": COSINE_THRESHOLD,
        "matched_face_box": best_box if matched else None,
        "total_faces":      len(faces),
        "reason":           "Match found" if matched else
                            f"Best cosine {best_cosine:.4f} < threshold {COSINE_THRESHOLD}",
    }