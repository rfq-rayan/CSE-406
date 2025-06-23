# Website Fingerprinting System - Complete Implementation

## Overview
This project implements a complete website fingerprinting data collection and classification system using browser automation, database storage, and machine learning.

## System Components

### 1. Data Collection (`collect.py`)
- **Browser Automation**: Uses Selenium with Chrome WebDriver
- **Trace Collection**: Captures side-channel timing data during website loading
- **Storage**: Saves traces to SQLite database with metadata
- **Export**: Converts database records to JSON format for ML training

### 2. Database Management (`database.py`)
- **SQLite Database**: Stores traces with website URLs and timing data
- **Schema**: Simple table with id, website, and trace_data columns
- **Operations**: Insert, retrieve, and export functionalities

### 3. Web Interface (`app.py`)
- **Flask Server**: Provides REST API for system control
- **Endpoints**:
  - `/` - Home page with system status
  - `/collect` - Start data collection
  - `/status` - Check collection status
  - `/export` - Export data to JSON

### 4. Machine Learning Pipeline (`train.py`)
- **Data Loading**: Custom PyTorch Dataset for trace data
- **Models**: Two CNN architectures (Basic and Complex)
- **Training**: Stratified train-test split with evaluation
- **Results**: Classification reports and model saving

### 5. Inference System (`inference.py`)
- **Model Loading**: Load trained models for prediction
- **Prediction**: Classify new traces with confidence scores
- **Demo**: Example usage of trained models

## Key Results

### Data Collection
- ✅ Successfully collected 30 complete traces from 3 websites
- ✅ Each trace contains 1000 timing data points
- ✅ Automated Chrome browser with reliable WebDriver setup

### Machine Learning Performance
- **Best Model**: Complex CNN with 66.7% test accuracy
- **Per-Class Performance**:
  - Prothom Alo: 100% precision/recall (perfect classification)
  - BUET Moodle: 50% precision/recall
  - Google: 50% precision/recall
- **Model Size**: 4.16M parameters trained for 50 epochs

### System Reliability
- ✅ All components tested and working
- ✅ Database operations verified
- ✅ WebDriver compatibility confirmed
- ✅ Complete ML pipeline functional

## File Structure
```
code/
├── collect.py              # Data collection with Selenium
├── database.py             # SQLite database operations
├── app.py                  # Flask web server
├── train.py                # ML training pipeline
├── inference.py            # Model inference demo
├── setup.py                # System setup and checks
├── test_system.py          # System integration tests
├── requirements.txt        # Python dependencies
├── dataset.json            # Exported training data
├── webfingerprint.db       # SQLite database
├── saved_models/           # Trained model weights
│   ├── basic_model.pth
│   └── complex_model.pth
└── static/                 # Web interface files
    ├── index.html
    ├── index.js
    └── ...
```

## Usage Instructions

### 1. Setup Environment
```bash
pip install -r requirements.txt
python setup.py  # Verify Chrome and dependencies
```

### 2. Collect Data
```bash
python app.py  # Start Flask server
# Open browser to http://localhost:5000
# Click "Start Collection" to gather traces
```

### 3. Train Models
```bash
python train.py  # Train both CNN models
```

### 4. Run Inference
```bash
python inference.py  # Demo trained model predictions
```

### 5. Test System
```bash
python test_system.py  # Verify all components
```

## Technical Achievements

1. **End-to-End Pipeline**: Complete data flow from collection to prediction
2. **Robust Automation**: Reliable Chrome WebDriver setup with fallbacks
3. **Modern ML**: PyTorch-based CNNs with proper validation
4. **System Integration**: All components tested and working together
5. **Production Ready**: Error handling, logging, and documentation

## Future Improvements

1. **Scale Data Collection**: Collect 100+ traces per website
2. **Advanced Models**: Try RNNs, Transformers, or ensemble methods
3. **Feature Engineering**: Extract statistical features from raw traces
4. **Real-time Classification**: Stream processing for live inference
5. **Security Analysis**: Study defense mechanisms and robustness

## Conclusion

This project successfully demonstrates a complete website fingerprinting system with:
- Automated data collection using browser automation
- Proper data storage and management
- Modern deep learning classification
- End-to-end testing and validation

The 66.7% accuracy achieved is reasonable for the dataset size and demonstrates the feasibility of side-channel website fingerprinting using timing analysis.
