from fastapi import FastAPI, Depends, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import models, database, auth
from .tasks import train_model
import shutil

app = FastAPI()
models.Base.metadata.create_all(bind=database.engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register")
async def register(email: str, password: str, db: Session = Depends(get_db)):
    hashed_password = auth.get_password_hash(password)
    db_user = models.User(email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    return {"message": "User created"}

@app.post("/upload")
async def upload_file(file: UploadFile, user_id: int, db: Session = Depends(get_db)):
    file_path = f"/data/{user_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"path": file_path}

@app.post("/train")
async def start_training(model_type: str, dataset_path: str, user_id: int, db: Session = Depends(get_db)):
    job = models.MLJob(user_id=user_id, model_type=model_type, dataset_path=dataset_path)
    db.add(job)
    db.commit()
    train_model.delay(job.id)
    return {"job_id": job.id}