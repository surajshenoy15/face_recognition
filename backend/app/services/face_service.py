"""
face_service.py - Core face recognition logic using YuNet + SFace (Apache 2.0 licensed)
Place this in: backend/app/services/face_service.py
"""

import cv2
import numpy as np
import base64
import json
from pathlib import Path
from typing import Optional

# ── Paths to your ONNX models (already in your project) ─────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models" / "face"
DETECTOR_MODEL = str(MODEL_DIR / "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = str(MODEL_DIR / "face_recognition_sface_2021dec.onnx")

# ── Similarity threshold (cosine distance) ───────────────────────────────────
# SFace cosine: >= 0.363 = same person (from OpenCV docs)
COSINE_THRESHOLD = 0.363
# L2 threshold: <= 1.128
L2_THRESHOLD = 1.128


class FaceService:
    def __init__(self):
        self._detector = None
        self._recognizer = None

    def _get_detector(self, input_width: int = 320, input_height: int = 320):
        """Lazy-load YuNet face detector."""
        self._detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL,
            "",
            (input_width, input_height),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        return self._detector

    def _get_recognizer(self):
        """Lazy-load SFace recognizer."""
        if self._recognizer is None:
            self._recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        return self._recognizer

    def _decode_image(self, image_b64: str) -> np.ndarray:
        """Decode base64 image string to OpenCV BGR array."""
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Ensure it's a valid JPEG/PNG.")
        return img

    def _detect_faces(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Run YuNet detection. Returns face boxes or None."""
        h, w = img.shape[:2]
        detector = self._get_detector(w, h)
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        return faces

    def _get_largest_face(self, faces: np.ndarray) -> np.ndarray:
        """Return the largest detected face (by area)."""
        if faces is None or len(faces) == 0:
            return None
        # face box: [x, y, w, h, ...] — sort by area descending
        areas = faces[:, 2] * faces[:, 3]
        return faces[np.argmax(areas)]

    def extract_embedding(self, image_b64: str) -> dict:
        """
        Extract 128-d face embedding from a single image.
        Returns: {"embedding": [...], "face_box": [x,y,w,h]} or raises.
        """
        img = self._decode_image(image_b64)
        faces = self._detect_faces(img)

        if faces is None or len(faces) == 0:
            raise ValueError("No face detected in image. Please retake with your face clearly visible.")

        face = self._get_largest_face(faces)
        recognizer = self._get_recognizer()
        aligned = recognizer.alignCrop(img, face)
        embedding = recognizer.feature(aligned)  # shape: (1, 128)
        embedding = embedding.flatten().tolist()

        box = face[:4].astype(int).tolist()  # [x, y, w, h]
        return {"embedding": embedding, "face_box": box}

    def average_embeddings(self, embeddings: list[list[float]]) -> list[float]:
        """Average multiple embeddings into one representative vector (L2 normalized)."""
        arr = np.array(embeddings, dtype=np.float32)
        avg = arr.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg.tolist()

    def match_face_in_group(
        self,
        group_image_b64: str,
        stored_embedding: list[float],
    ) -> dict:
        """
        Detect all faces in a group photo and find the best match
        against the stored student embedding.
        Returns match result with confidence score.
        """
        img = self._decode_image(group_image_b64)
        faces = self._detect_faces(img)

        if faces is None or len(faces) == 0:
            return {
                "matched": False,
                "reason": "No faces detected in group photo.",
                "best_score": None,
                "matched_face_box": None,
                "total_faces": 0,
            }

        recognizer = self._get_recognizer()
        stored_emb = np.array(stored_embedding, dtype=np.float32).reshape(1, -1)

        best_cosine = -1.0
        best_l2 = float("inf")
        best_box = None

        for face in faces:
            try:
                aligned = recognizer.alignCrop(img, face)
                emb = recognizer.feature(aligned)
                cosine_score = recognizer.match(stored_emb, emb, cv2.FaceRecognizerSF_FR_COSINE)
                l2_score = recognizer.match(stored_emb, emb, cv2.FaceRecognizerSF_FR_NORM_L2)

                if cosine_score > best_cosine:
                    best_cosine = cosine_score
                    best_l2 = l2_score
                    best_box = face[:4].astype(int).tolist()
            except Exception:
                continue

        matched = best_cosine >= COSINE_THRESHOLD and best_l2 <= L2_THRESHOLD

        return {
            "matched": matched,
            "cosine_score": round(float(best_cosine), 4),
            "l2_score": round(float(best_l2), 4),
            "cosine_threshold": COSINE_THRESHOLD,
            "matched_face_box": best_box if matched else None,
            "total_faces": len(faces),
            "reason": "Match found" if matched else f"Best cosine score {best_cosine:.4f} below threshold {COSINE_THRESHOLD}",
        }


# Singleton instance
face_service = FaceService()