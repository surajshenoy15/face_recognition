from fastapi import FastAPI
from app.routes.student_face_routes import router as face_router

app = FastAPI(
    title="Face Recognition API",
    description="Student face enrollment and activity verification using YuNet + SFace",
    version="1.0.0",
)

app.include_router(face_router)

@app.get("/")
def root():
    return {"status": "running", "docs": "/docs"}