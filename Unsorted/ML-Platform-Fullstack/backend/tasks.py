from celery import Celery
from . import models, database
import mlflow

celery = Celery(__name__, broker="redis://redis:6379/0")

@celery.task
def train_model(job_id):
    db = database.SessionLocal()
    job = db.query(models.MLJob).filter(models.MLJob.id == job_id).first()
    job.status = "training"
    db.commit()

    try:
        mlflow.set_experiment("default")
        with mlflow.start_run():
            # Add actual training logic here
            mlflow.log_param("model_type", job.model_type)
            mlflow.log_metric("accuracy", 0.95)
            job.status = "completed"
    except Exception as e:
        job.status = "failed"
    
    db.commit()
    db.close()