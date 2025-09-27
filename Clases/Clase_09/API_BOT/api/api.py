from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import joblib
import numpy as np
import xgboost as xgb
import warnings
from typing import List

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI(title="Fetal Health Prediction API", version="1.0.0")

try:
    scaler = joblib.load("scaler_min_max.joblib")
    
    model = xgb.XGBClassifier()
    model.load_model("model_xgb.json")
except FileNotFoundError as e:
    raise RuntimeError("Model or scaler file not found. Ensure the files are in the correct path.") from e


class PredictionInput(BaseModel):
    baseline_value: float
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    severe_decelerations: float
    prolongued_decelerations: float
    abnormal_short_term_variability: float
    mean_value_of_short_term_variability: float
    percentage_of_time_with_abnormal_long_term_variability: float
    mean_value_of_long_term_variability: float
    histogram_width: float
    histogram_min: float
    histogram_max: float
    histogram_number_of_peaks: float
    histogram_number_of_zeroes: float
    histogram_mean: float
    histogram_variance: float
    histogram_tendency: float

class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float

class InfoResponde(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ml_model: dict
    features: List[str]
    target_classes: dict

@app.get("/")
async def root():
    """ Endpoint de bienvenida """
    return {"message": "Welcome to the Fetal Health Prediction API. Visit /docs for API documentation."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """ Recibimos datos, procesamos, y devolvemos """
    try:

        features = np.array(list(input_data.model_dump().values())).reshape(1, -1)
        
        features_scaled = scaler.transform(features)

        prediction_raw = model.predict(features_scaled)
        prediction =  int(prediction_raw[0]) + 1

        prediction_proba = model.predict_proba(features_scaled)
        confidence = float(np.max(prediction_proba))

        prediction_labels = {
            1: "Normal",
            2: "Suspect",
            3: "Pathological"
        }

        return PredictionOutput(
            prediction=prediction,
            prediction_label=prediction_labels.get(prediction, "Unknown"),
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/info", response_model=InfoResponde)
async def get_info():
    """ Endpoint para obtener información del modelo """
    try:
        info = {
            "ml_model": {
                "type": "XGBoost Classifier",
                "version": xgb.__version__
            },
            "features": [
                "baseline_value", "accelerations", "fetal_movement", "uterine_contractions",
                "light_decelerations", "severe_decelerations", "prolongued_decelerations",
                "abnormal_short_term_variability", "mean_value_of_short_term_variability",
                "percentage_of_time_with_abnormal_long_term_variability", "mean_value_of_long_term_variability",
                "histogram_width", "histogram_min", "histogram_max", "histogram_number_of_peaks",
                "histogram_number_of_zeroes", "histogram_mean", "histogram_variance", "histogram_tendency"
            ],
            "target_classes": {
                1: "Normal",
                2: "Suspect",
                3: "Pathological"
            }
        }
        return InfoResponde(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

