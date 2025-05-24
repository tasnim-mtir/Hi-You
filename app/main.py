from fastapi import APIRouter, HTTPException
from .models import train_model
from .schemas import TrainRequest, TrainResponse, PredictRequest, PredictResponse
import joblib
import pandas as pd

router = APIRouter(prefix="/api")

@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    try:
        result = train_model(request.data_path, request.model_save_path)
        return {
            "message": "Model trained successfully",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        model = joblib.load(request.model_path)
        
        # If feature names are provided, ensure correct order
        if request.feature_names_path:
            feature_names = joblib.load(request.feature_names_path)
            input_df = pd.DataFrame([request.features], columns=feature_names)
            prediction = model.predict(input_df)
        else:
            prediction = model.predict([request.features])
        
        return {
            "prediction": prediction.tolist(),
            "result_label": "Present" if prediction[0] == 1 else "Absent"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))