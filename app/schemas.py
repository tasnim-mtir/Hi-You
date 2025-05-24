from pydantic import BaseModel
from typing import List, Optional

class TrainRequest(BaseModel):
    data_path: str
    model_save_path: str
    feature_names_path: Optional[str] = None

class TrainResponse(BaseModel):
    message: str
    result: dict

class PredictRequest(BaseModel):
    model_path: str
    features: List[int]
    feature_names_path: Optional[str] = None

class PredictResponse(BaseModel):
    prediction: List[int]
    result_label: str