# Enhanced ECG App Documentation

## Overview

The `enhanced_ecg_app.py` is an advanced Streamlit application for comprehensive ECG analysis with a focus on atrial fibrillation detection, feature extraction, and interactive visualization. It combines traditional signal processing techniques with machine learning approaches to provide detailed ECG analysis.

## Features

- **Advanced ECG Analysis**: Comprehensive analysis with 100+ cardiac biomarkers
- **Multi-Classifier Approach**: Combines traditional ML and deep learning models (when available)
- **Interactive Visualizations**: Rich, annotated visualizations of ECG signals
- **Feature Extraction**: Detailed extraction and display of ECG features
- **Analysis Logging**: Keeps track of analyses for comparison and tracking
- **Data Export**: Export capabilities for further analysis
- **Signal Issue Detection**: Automatically detects and fixes common signal problems

## Dependencies

- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data handling and numerical operations
- **Matplotlib/Plotly**: Visualization libraries
- **SciPy**: Signal processing functions
- **Custom Modules**:
  - `ecg_holter_analysis.py`: Contains the `HolterAnalyzer` class
  - `ecg_arrhythmia_classification.py`: Contains the `ECGArrhythmiaClassifier` class
  - `ecg_medical_analysis.py`: Contains the `ECGMedicalAnalysis` class
  - `ecg_advanced_features.py`: Contains the `ECGFeatureExtractor` class
  - `ecg_multi_classifier.py`: Contains the `ECGMultiClassifier` class (optional)

## Key Functions

### `fix_signal_issues(signal)`
Fixes common issues with ECG signals that cause visualization problems.

**Parameters:**
- `signal`: ECG signal array

**Returns:**
- Fixed signal array

### `plot_ecg(df, title="ECG Signal", use_plotly=False)`
Plots ECG signal using either Matplotlib or Plotly.

**Parameters:**
- `df`: DataFrame with 'time' and 'signal' columns
- `title`: Plot title (default: "ECG Signal")
- `use_plotly`: Whether to use Plotly for plotting (default: False)

**Returns:**
- Figure object (Matplotlib or Plotly)

### `plot_ecg_with_peaks(df, r_peaks=None, title="ECG Signal with R-peaks", use_plotly=False)`
Plots ECG signal with R-peaks marked.

**Parameters:**
- `df`: DataFrame with 'time' and 'signal' columns
- `r_peaks`: Indices of R-peaks (default: None)
- `title`: Plot title (default: "ECG Signal with R-peaks")
- `use_plotly`: Whether to use Plotly for plotting (default: False)

**Returns:**
- Figure object (Matplotlib or Plotly)

### `display_raw_data(df, max_rows=20)`
Displays raw data table with signal values.

**Parameters:**
- `df`: DataFrame with 'time' and 'signal' columns
- `max_rows`: Maximum number of rows to display (default: 20)

### `display_features(features)`
Displays extracted features in organized categories.

**Parameters:**
- `features`: Dictionary of extracted features

### `analyze_af_segment(signal_data, sampling_rate=200, include_features=True)`
Analyzes a segment for AF detection with enhanced features.

**Parameters:**
- `signal_data`: ECG signal data
- `sampling_rate`: Sampling rate in Hz (default: 200)
- `include_features`: Whether to include detailed feature extraction (default: True)

### `plot_ecg_with_annotations(df, r_peaks=None, af_prob=None, analysis_regions=None, heart_rate=None, title="ECG Signal with Analysis")`
Creates an interactive ECG visualization with AI analysis annotations.

**Parameters:**
- `df`: DataFrame with 'time' and 'signal' columns
- `r_peaks`: Indices of R-peaks (default: None)
- `af_prob`: Atrial Fibrillation probability (0-1) (default: None)
- `analysis_regions`: Regions of interest in the signal (default: None)
- `heart_rate`: Heart rate value (default: None)
- `title`: Plot title (default: "ECG Signal with Analysis")

**Returns:**
- Plotly figure object

### `main()`
Main function that sets up the Streamlit interface with multiple tabs for different analysis features.

## Application Structure

The application is organized into three main tabs:

1. **ECG Analysis**: For uploading, analyzing, and visualizing ECG data
2. **Analysis Log**: Tracks and compares multiple analyses
3. **About This Tool**: Information about the application and its features

## Advanced Analysis Features

- **Feature Extraction**: Extracts 100+ cardiac biomarkers from ECG signals
- **Multi-Classifier Approach**: Uses multiple classification methods when available
- **TERMA-inspired R-peak Detection**: More accurate peak detection based on research
- **Explainable AI**: Uses SHAP values to explain model predictions (when available)
- **Analysis Logging**: Keeps track of analyses for comparison and tracking
- **Correlation Analysis**: Shows relationships between different ECG metrics

## Signal Processing Techniques

The application employs several signal processing techniques:

- **Bandpass Filtering**: Removes noise and baseline wander
- **R-peak Detection**: Identifies heart beats
- **Heart Rate Variability Analysis**: Measures variations in heart rate
- **Frequency Domain Analysis**: Examines frequency components of the ECG
- **Time Domain Analysis**: Analyzes temporal features of the ECG

## Usage Example

```python
# Run the Streamlit app
streamlit run enhanced_ecg_app.py
```

Then upload ECG data through the web interface to begin analysis.

## Data Handling

- **File Upload**: Supports various file formats including CSV and EDF
- **Signal Preprocessing**: Automatically cleans and prepares signals
- **Feature Extraction**: Extracts comprehensive set of features
- **Data Export**: Allows exporting of analysis results

## Visualization Features

- **Interactive Plots**: Zoom, pan, and hover capabilities
- **Annotated ECG**: Shows R-peaks, AF probability, and other metrics
- **Feature Visualization**: Displays extracted features in organized categories
- **Comparison Visualizations**: Compares analyses across multiple files

## Notes

- The application is designed for research and educational purposes
- The multi-classifier approach requires the optional `ecg_multi_classifier.py` module
- Advanced features may require significant computational resources
- The application includes detailed explanations of metrics and analyses for educational value
