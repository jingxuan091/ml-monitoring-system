"""
Production FastAPI application with monitoring
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict
import json
import joblib
from glob import glob

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API with Monitoring",
    description="Production fraud detection with real-time monitoring and drift detection",
    version="2.0.0"
)

# Global variables for model and monitoring
model = None
model_info = None
predictions_log = []
performance_metrics = []

def load_best_model():
    """Load the highest performing model"""
    global model, model_info
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        return False
    
    # Find the final tuned model
    final_models = glob(os.path.join(models_dir, "final_tuned_rf_*.joblib"))
    
    if final_models:
        latest_model = sorted(final_models)[-1]
        model = joblib.load(latest_model)
        
        model_info = {
            "path": latest_model,
            "type": "Random Forest (Final Tuned)",
            "performance": "AUC: 0.9751",
            "loaded_at": datetime.now().isoformat()
        }
        return True
    
    return False

# Load model on startup
load_best_model()

# Request/Response models
class TransactionRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    model_version: str
    timestamp: str
    confidence: float

class MonitoringStats(BaseModel):
    total_predictions: int
    fraud_rate: float
    avg_probability: float
    high_risk_count: int
    model_performance: Dict

@app.get("/")
def read_root():
    return {
        "message": "Fraud Detection API with Monitoring",
        "model_loaded": model is not None,
        "model_info": model_info
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": model_info,
        "predictions_count": len(predictions_log),
        "uptime": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    """Predict fraud with monitoring"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        df = pd.DataFrame([transaction.dict()])[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        # Calculate confidence and risk level
        confidence = max(model.predict_proba(df)[0])
        
        if probability >= 0.8:
            risk_level = "HIGH"
        elif probability >= 0.5:
            risk_level = "MEDIUM"
        elif probability >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Create response
        response = PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            model_version=model_info["path"].split('/')[-1] if model_info else "unknown",
            timestamp=datetime.now().isoformat(),
            confidence=float(confidence)
        )
        
        # Log for monitoring
        log_entry = {
            "timestamp": response.timestamp,
            "prediction": response.prediction,
            "probability": response.probability,
            "risk_level": response.risk_level,
            "confidence": response.confidence,
            "features": transaction.dict()
        }
        predictions_log.append(log_entry)
        
        # Keep only last 10000 predictions in memory
        if len(predictions_log) > 10000:
            predictions_log.pop(0)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/monitoring/stats", response_model=MonitoringStats)
def get_monitoring_stats():
    """Get current monitoring statistics"""
    
    if not predictions_log:
        return MonitoringStats(
            total_predictions=0,
            fraud_rate=0.0,
            avg_probability=0.0,
            high_risk_count=0,
            model_performance={}
        )
    
    # Calculate statistics
    total_predictions = len(predictions_log)
    fraud_predictions = sum(1 for p in predictions_log if p["prediction"] == 1)
    fraud_rate = fraud_predictions / total_predictions
    
    avg_probability = np.mean([p["probability"] for p in predictions_log])
    high_risk_count = sum(1 for p in predictions_log if p["risk_level"] == "HIGH")
    
    # Recent performance (last 1000 predictions)
    recent_predictions = predictions_log[-1000:] if len(predictions_log) > 1000 else predictions_log
    recent_fraud_rate = sum(1 for p in recent_predictions if p["prediction"] == 1) / len(recent_predictions)
    recent_avg_prob = np.mean([p["probability"] for p in recent_predictions])
    
    return MonitoringStats(
        total_predictions=total_predictions,
        fraud_rate=fraud_rate,
        avg_probability=avg_probability,
        high_risk_count=high_risk_count,
        model_performance={
            "baseline_auc": 0.9751,
            "recent_fraud_rate": recent_fraud_rate,
            "recent_avg_probability": recent_avg_prob,
            "model_info": model_info
        }
    )

@app.get("/monitoring/predictions")
def get_recent_predictions():
    """Get recent predictions for analysis"""
    return {
        "total_count": len(predictions_log),
        "recent_predictions": predictions_log[-100:],  # Last 100
        "summary": {
            "fraud_rate": sum(1 for p in predictions_log if p["prediction"] == 1) / len(predictions_log) if predictions_log else 0,
            "avg_risk_score": np.mean([p["probability"] for p in predictions_log]) if predictions_log else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
