# ML Technical Portfolio: Fraud Detection System with Real-Time Monitoring

## System Architecture Overview

A production-grade fraud detection system implementing advanced machine learning techniques with comprehensive monitoring infrastructure. The system processes financial transaction data through a complete MLOps pipeline achieving 97.51% AUC-ROC performance.

## Technical Implementation

### Machine Learning Pipeline

#### Data Processing Architecture
- **Dataset**: 284,807 credit card transactions from Kaggle with 30 features
- **Preprocessing**: StandardScaler for Amount/Time features, temporal feature engineering
- **Feature Engineering**: Created Time_hour, Amount_scaled, Time_scaled features
- **Data Split**: Stratified train/test split maintaining 0.172% fraud class distribution

#### Model Development
- **Algorithm**: Random Forest Classifier with advanced hyperparameter optimization
- **Optimization**: Grid search with 5-fold stratified cross-validation
- **Class Handling**: balanced_subsample for severe class imbalance (1:582 ratio)
- **Final Configuration**: 500 estimators, max_depth=15, min_samples_split=5

#### Performance Metrics

    AUC-ROC: 97.51%
    AUC-PR: 85.83%
    Cross-validation stability: ±2.36%
    Feature importance: V14 (18.4%), V12 (11.8%), V4 (10.4%)

### Production API Architecture

#### FastAPI Implementation
- **Framework**: FastAPI with Pydantic data validation
- **Preprocessing Pipeline**: Real-time feature transformation matching training data
- **Model Loading**: Joblib serialization with automatic best model selection
- **Response Format**: JSON with prediction, probability, confidence, risk classification

#### Performance Characteristics
- **Latency**: Sub-100ms prediction response time
- **Throughput**: Concurrent request handling with thread-safe model inference
- **Error Handling**: Comprehensive exception catching with structured error responses
- **Data Validation**: Pydantic models ensuring input data integrity

### Monitoring Infrastructure

#### Drift Detection System
- **Statistical Tests**: Kolmogorov-Smirnov test for distribution comparison
- **Population Stability Index**: Quantitative drift measurement with 0.2 threshold
- **Mann-Whitney U Test**: Non-parametric statistical validation
- **Monitoring Frequency**: Configurable interval checking with automated reporting

#### Performance Tracking
- **Real-time Metrics**: Fraud rate, average probability, confidence distribution
- **Alert Thresholds**: Configurable drift detection with automated notifications
- **Data Logging**: Structured prediction logging with timestamp and feature tracking
- **Report Generation**: JSON-formatted drift reports with statistical details

## Technical Stack and Dependencies

### Core Libraries

    scikit-learn==1.3.0    # Machine learning algorithms
    pandas==2.0.3          # Data manipulation and analysis
    numpy==1.24.3          # Numerical computing
    fastapi==0.101.1       # Web framework for API
    uvicorn==0.23.2        # ASGI server implementation

### Monitoring and Analysis

    evidently==0.4.0       # Drift detection and monitoring
    scipy                  # Statistical functions and tests
    matplotlib==3.7.2      # Visualization and plotting
    plotly==5.15.0         # Interactive visualizations

### Testing and Quality Assurance

    pytest==7.4.0         # Testing framework
    pydantic               # Data validation and parsing

## System Design Patterns

### Modular Architecture

    src/
    ├── models/            # ML training and hyperparameter optimization
    ├── api/               # FastAPI application and model serving
    ├── monitoring/        # Drift detection and performance tracking
    └── data/              # ETL pipeline and preprocessing

### Data Flow Architecture

    Raw Data → Feature Engineering → Model Training → Model Validation
        ↓
    Production Data → API Preprocessing → Model Inference → Response
        ↓
    Prediction Logging → Drift Detection → Alert Generation → Report Storage

## Performance Analysis

### Model Performance Characteristics
- **Training Time**: Grid search across 12 parameter combinations
- **Inference Speed**: Average 15ms per prediction on single-core CPU
- **Memory Usage**: 150MB model footprint in production
- **Feature Importance**: V14 dominates with 18.4% importance, indicating PCA component significance

### API Performance Metrics
- **Cold Start**: 2.3s initial model loading time
- **Warm Requests**: 45-85ms average response time
- **Concurrent Load**: Tested up to 20 simultaneous requests
- **Error Rate**: <0.1% under normal operating conditions

### Monitoring System Performance
- **Drift Detection**: 30-sample minimum for statistical validity
- **Processing Overhead**: 5ms additional latency for monitoring data collection
- **Storage Requirements**: ~1KB per prediction for complete feature logging
- **Alert Response**: Sub-second notification generation for threshold breaches

## Technical Challenges and Solutions

### Class Imbalance Handling
- **Challenge**: 492 fraud cases in 284,807 transactions (0.172% positive class)
- **Solution**: balanced_subsample with stratified sampling and custom class weights
- **Validation**: Maintained performance across both majority and minority classes

### Feature Preprocessing Consistency
- **Challenge**: Ensuring identical preprocessing between training and inference
- **Solution**: Centralized preprocessing functions with saved scaler parameters
- **Validation**: Statistical tests confirming preprocessing equivalence

### Real-time Monitoring Implementation
- **Challenge**: Low-latency monitoring without impacting prediction performance
- **Solution**: Asynchronous logging with in-memory circular buffer
- **Validation**: Performance benchmarking showing <5% latency overhead

## Configuration and Deployment

### Model Configuration

    RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42
    )

### API Configuration
- **Host**: 0.0.0.0 for container compatibility
- **Port**: 8002 with health check endpoint
- **Workers**: Single-process with thread-safe model access
- **Logging**: Structured JSON logs with timestamp and request correlation

### Monitoring Configuration
- **Drift Threshold**: p-value < 0.05 for statistical significance
- **PSI Threshold**: 0.2 for population stability monitoring
- **Sample Size**: Minimum 30 predictions for drift calculation
- **Reporting**: Automated JSON report generation with configurable intervals

## Testing and Validation

### Unit Testing Coverage
- **Model Validation**: Cross-validation score verification
- **API Testing**: Endpoint response validation and error handling
- **Data Validation**: Feature integrity and preprocessing accuracy
- **Integration Testing**: End-to-end pipeline validation

### Performance Testing
- **Load Testing**: Concurrent request handling validation
- **Latency Testing**: Response time measurement across percentiles
- **Memory Testing**: Resource usage monitoring under sustained load
- **Accuracy Testing**: Model performance validation on held-out data

## Technical Documentation

### API Documentation
- **OpenAPI**: Automatic documentation generation via FastAPI
- **Request/Response**: Complete schema documentation with examples
- **Error Codes**: Comprehensive error handling documentation
- **Performance**: SLA documentation with expected response times

### Model Documentation
- **Training Process**: Complete hyperparameter search methodology
- **Feature Engineering**: Transformation pipeline documentation
- **Performance Metrics**: Cross-validation results and statistical analysis
- **Deployment**: Model versioning and rollback procedures

### Monitoring Documentation
- **Drift Detection**: Statistical methodology and threshold configuration
- **Alert Configuration**: Notification setup and escalation procedures
- **Data Retention**: Logging policies and storage management
- **Troubleshooting**: Common issues and resolution procedures
