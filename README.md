# ML Model Monitoring System

A production-ready fraud detection system with real-time monitoring capabilities, achieving **97.51% AUC** performance on the Kaggle Credit Card Fraud Detection dataset.

## 🎯 Overview

This project demonstrates end-to-end MLOps capabilities including model training, hyperparameter optimization, API serving, and real-time monitoring. The system processes credit card transactions and provides fraud risk assessments with comprehensive monitoring.

## ✨ Key Features

- **High-Performance Model**: Random Forest achieving 97.51% AUC
- **Production API**: FastAPI with real-time prediction serving  
- **Real-Time Monitoring**: Live dashboard with risk assessment
- **Data Processing**: Complete pipeline from raw data to production
- **Drift Detection**: Statistical monitoring for model degradation

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | **97.51%** |
| AUC-PR | 85.83% |
| Dataset Size | 284,807 transactions |
| Fraud Rate | 0.172% |
| API Response Time | <100ms |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment

### Installation
```bash
# Clone repository
git clone <repository-url>
cd ml-monitoring-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
