from fastapi import FastAPI

app = FastAPI(
    title="Lung Cancer Prediction API",
    description="API for predicting lung cancer risk using Gradient Boosting",
    version="1.0.0"
)

# Import routes after app creation
from .main import router
app.include_router(router)