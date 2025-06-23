# Real-Time Website Detection - Side-Channel Attack Demo

This folder contains a powerful demonstration of real-time website fingerprinting using side-channel attacks and machine learning.

## 🎯 What This Demo Does

This Flask application demonstrates how an attacker could potentially identify which website a user is visiting in real-time by analyzing side-channel timing data from the browser. It showcases the privacy implications of website fingerprinting attacks.

### Key Features

- **Real-Time Detection**: Analyzes browsing patterns in real-time
- **Machine Learning**: Uses trained CNN models for classification
- **Interactive Web Interface**: Beautiful, modern UI for demonstrations
- **Side-Channel Analysis**: Collects timing data using browser APIs
- **Demo Mode**: Test with sample data for educational purposes

## 🔬 How It Works

1. **Timing Collection**: The app collects side-channel timing data from browser performance APIs
2. **Pattern Analysis**: A trained Convolutional Neural Network analyzes the timing patterns
3. **Website Fingerprinting**: The model identifies unique timing signatures for different websites
4. **Real-Time Classification**: Predicts which website the user is likely visiting

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Chrome browser
- Trained ML models (from main project)

### Setup
```bash
cd realtime_detection
python setup.py        # Check dependencies and setup
python app.py          # Start the demo app
```

Then open your browser to: **http://localhost:5001**

## 📱 Web Interface

The interface provides:

### Live Detection Panel
- **Start Analysis**: Begin real-time side-channel data collection
- **Status Display**: Shows current analysis status with animations
- **Prediction Results**: Displays detected website with confidence score
- **Visual Confidence**: Animated progress bar showing prediction confidence

### Demo Panel
- **Sample Testing**: Test the model with known good data
- **Accuracy Display**: Shows if predictions match expected results
- **Educational Tool**: Safe way to understand the technology

### Information Panel
- **How It Works**: Detailed explanation of the attack methodology
- **Model Details**: Technical specifications of the neural network
- **Performance Metrics**: Speed and accuracy information

## 🛡️ Ethical Considerations

### ⚠️ Educational Purpose Only
This demonstration is intended for:
- **Security Research**: Understanding attack vectors
- **Privacy Education**: Showing potential vulnerabilities
- **Academic Study**: Learning about side-channel attacks
- **Defense Development**: Building countermeasures

### 🔒 Responsible Use
- **Do not use for malicious purposes**
- **Respect user privacy and consent**
- **Follow applicable laws and regulations**
- **Use only for authorized testing**

## 🔧 Technical Details

### Architecture
- **Backend**: Flask web server with real-time prediction
- **Frontend**: Modern HTML5/CSS3/JavaScript interface
- **ML Model**: PyTorch CNN with 66.7% accuracy
- **Data Collection**: Selenium WebDriver for timing analysis

### Model Performance
- **Accuracy**: 66.7% on test dataset
- **Model Size**: 4.16M parameters
- **Prediction Time**: <1 second
- **Collection Time**: ~10 seconds

### Supported Websites
Currently trained to detect:
- BUET CSE Moodle (https://cse.buet.ac.bd/moodle/)
- Google (https://google.com)
- Prothom Alo (https://prothomalo.com)

## 📊 Results and Limitations

### Strengths
- ✅ Real-time classification capability
- ✅ User-friendly demonstration interface
- ✅ Educational value for security awareness
- ✅ Modular design for easy extension

### Current Limitations
- 📊 Limited training data (30 traces from 3 websites)
- 🎯 Moderate accuracy (66.7% overall)
- 🌐 Small number of target websites
- ⚡ Requires specific browser conditions

### Potential Improvements
- 📈 Collect more training data (100+ traces per site)
- 🌍 Add support for more websites
- 🔧 Implement data augmentation techniques
- 🛡️ Add defense mechanism demonstrations

## 🗂️ File Structure

```
realtime_detection/
├── app.py                  # Main Flask application
├── models.py               # Neural network model definitions
├── setup.py                # Setup and dependency checker
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
└── templates/
    └── realtime.html      # Web interface template
```

## 🔗 Integration with Main Project

This real-time detection app integrates with the main website fingerprinting project:

- **Uses trained models** from `../code/saved_models/`
- **Reads dataset metadata** from `../code/dataset.json`
- **Leverages ML infrastructure** built in the main project
- **Demonstrates practical application** of the trained models

## 🎥 Demo Instructions

1. **Start the app**: Run `python app.py`
2. **Open browser**: Navigate to http://localhost:5001
3. **Try demo mode**: Click "Run Demo with Sample Data" for safe testing
4. **Live detection**: Click "Start Side-Channel Analysis" for real-time demo
5. **Observe results**: Watch the prediction confidence and website classification

## 🔮 Future Enhancements

- **Multi-tab detection**: Analyze multiple browser tabs simultaneously
- **Real-time streaming**: Continuous monitoring and classification
- **Defense mechanisms**: Implement and test countermeasures
- **Mobile support**: Extend to mobile browser fingerprinting
- **Network analysis**: Include network timing patterns

## 📚 Educational Value

This demo serves as an excellent educational tool for:
- **Cybersecurity courses**: Demonstrating attack vectors
- **Privacy research**: Understanding fingerprinting techniques
- **Machine learning applications**: Showing practical ML use cases
- **Web security**: Illustrating browser-based vulnerabilities

---

**Remember**: This technology demonstrates serious privacy implications. Always use responsibly and ethically!
