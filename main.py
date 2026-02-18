from fastapi import FastAPI
from app.routes.face_routes import router as face_router
from app.database import engine, Base

# Create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Face Recognition API",
    description="Student face enrollment & activity verification — YuNet + SFace (Apache 2.0)",
    version="1.0.0",
)

app.include_router(face_router)

@app.get("/")
def root():
    return {"status": "running", "docs": "/docs"}