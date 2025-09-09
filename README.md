# ML Model Monitoring System

A production-ready fraud detection system with real-time monitoring capabilities, achieving **97.51% AUC** performance on the Kaggle Credit Card Fraud Detection dataset.

## Overview

This project demonstrates end-to-end MLOps capabilities including model training, hyperparameter optimization, API serving, and real-time monitoring. The system processes credit card transactions and provides fraud risk assessments with comprehensive monitoring.

## Key Features

- **High-Performance Model**: Random Forest achieving 97.51% AUC
- **Production API**: FastAPI with real-time prediction serving  
- **Real-Time Monitoring**: Live dashboard with risk assessment
- **Data Processing**: Complete pipeline from raw data to production
- **Drift Detection**: Statistical monitoring for model degradation

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | **97.51%** |
| AUC-PR | 85.83% |
| Dataset Size | 284,807 transactions |
| Fraud Rate | 0.172% |
| API Response Time | <100ms |

## Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment

### Installation

    # Clone repository
    git clone <repository-url>
    cd ml-monitoring-system

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt

### Data Setup
1. Download the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in `data/raw/`
3. Run preprocessing: `python src/data/improve_data.py`

### Model Training

    python src/models/final_tune.py

### Start API & Monitor

    # Start API server
    python src/api/working_main.py

    # In another terminal: Run monitoring
    python src/monitoring/live_dashboard.py

    # Run complete demo
    python portfolio_demo.py

## Project Structure

    ml-monitoring-system/
    ├── src/
    │   ├── models/           # Model training and optimization
    │   ├── api/              # Production API serving
    │   ├── monitoring/       # Drift detection and monitoring
    │   └── data/             # Data processing pipeline
    ├── data/
    │   ├── raw/              # Original dataset
    │   └── processed/        # Processed training/test data
    ├── models/               # Trained model artifacts
    └── requirements.txt      # Project dependencies

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System status and model info |
| `/predict` | POST | Single transaction prediction |
| `/stats` | GET | Monitoring metrics and statistics |

### Example API Usage

    import requests

    # Single prediction
    transaction = {
        "V1": -1.3598071, "V2": -0.0727812, # ... V3-V28
        "Time": 0, "Amount": 149.62
    }

    response = requests.post("http://localhost:8002/predict", json=transaction)
    print(response.json())
    # Output: {"prediction": 0, "probability": 0.0234, "risk_level": "LOW"}

## Monitoring Features

- **Real-time Statistics**: Fraud rate, average risk score, alert counts
- **Risk Assessment**: MINIMAL/LOW/MEDIUM/HIGH/CRITICAL classification
- **Performance Tracking**: Model accuracy and prediction confidence
- **Drift Detection**: Statistical tests for data distribution changes

## Model Development

The 97.51% AUC performance was achieved through:

1. **Feature Engineering**: Scaling Time/Amount features, creating time-based features
2. **Advanced Tuning**: Grid search with stratified cross-validation
3. **Class Balancing**: Using `balanced_subsample` for imbalanced data
4. **Model Selection**: Random Forest with 500 estimators, max_depth=15

## Technical Implementation

### MLOps Pipeline

    Data Ingestion → Feature Engineering → Model Training → Hyperparameter Tuning
           ↓                                                           ↑
    Model Serving ← API Deployment ← Model Validation ← Model Selection
           ↓
    Monitoring & Alerting ← Performance Tracking ← Prediction Logging

### Business Impact

This system addresses critical fraud detection needs:
- **Real-time Risk Assessment**: Sub-100ms prediction latency
- **High Accuracy**: 97.51% AUC meets industry standards
- **Scalable Architecture**: Production-ready API design
- **Operational Monitoring**: Live dashboards for rapid response

## Future Enhancements

- Docker containerization for deployment
- CI/CD pipeline with automated testing
- Advanced drift detection algorithms
- A/B testing framework for model versions
- Cloud deployment with auto-scaling

## License

This project is for portfolio demonstration purposes.
