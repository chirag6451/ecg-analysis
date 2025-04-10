# ECG Streamlit App Documentation

## Overview

The `ecg_streamlit_app.py` is a comprehensive web application built with Streamlit for analyzing ECG (Electrocardiogram) data from EDF files. It provides a user-friendly interface for uploading, visualizing, and analyzing ECG data with a focus on arrhythmia detection and medical reporting.

## Features

- **EDF File Upload**: Supports uploading and processing of EDF format ECG recordings
- **Holter Analysis**: Processes long-term ECG recordings (Holter monitor data)
- **Arrhythmia Detection**: Identifies various cardiac arrhythmias including atrial fibrillation
- **Medical Reporting**: Generates comprehensive medical reports with findings and metrics
- **Interactive Visualizations**: Displays ECG signals with annotations and markers
- **Downloadable Reports**: Provides HTML reports for download and sharing

## Dependencies

- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data handling and numerical operations
- **Matplotlib/Plotly**: Visualization libraries
- **NeuroKit2**: ECG signal processing
- **SciPy**: Signal processing functions
- **MNE**: EDF file reading
- **Custom Modules**:
  - `ecg_holter_analysis.py`: Contains the `HolterAnalyzer` class
  - `ecg_arrhythmia_classification.py`: Contains the `ECGArrhythmiaClassifier` class
  - `ecg_medical_analysis.py`: Contains the `ECGMedicalAnalysis` class

## Key Functions

### `plot_ecg(df, title="ECG Signal")`
Plots the ECG signal using Matplotlib.

**Parameters:**
- `df`: DataFrame with 'time' and 'signal' columns
- `title`: Plot title (default: "ECG Signal")

### `plot_ecg_with_peaks(df, r_peaks=None, title="ECG Signal")`
Plots the ECG signal with R-peaks marked.

**Parameters:**
- `df`: DataFrame with 'time' and 'signal' columns
- `r_peaks`: Indices of R-peaks (default: None)
- `title`: Plot title (default: "ECG Signal")

### `detect_arrhythmias(ecg_signal, fs)`
Detects arrhythmias in the ECG signal.

**Parameters:**
- `ecg_signal`: ECG signal array
- `fs`: Sampling frequency in Hz

**Returns:**
- Dictionary with arrhythmia information
- R-peaks indices
- RR intervals

### `plot_ecg_with_arrhythmia_markers(df, arrhythmias, rpeaks=None)`
Plots ECG with arrhythmia markers and annotations.

**Parameters:**
- `df`: DataFrame with 'time' and 'signal' columns
- `arrhythmias`: Dictionary with arrhythmia information
- `rpeaks`: Indices of R-peaks (default: None)

### `display_medical_report(report)`
Displays a medical analysis report in a user-friendly format.

**Parameters:**
- `report`: Dictionary containing medical report information

### `analyze_af_segment(signal_data, sampling_rate=200)`
Analyzes a segment for atrial fibrillation detection.

**Parameters:**
- `signal_data`: ECG signal data
- `sampling_rate`: Sampling rate in Hz (default: 200)

### `analyze_af_segment_minimal(signal_data, sampling_rate=200)`
Minimal version of the AF analysis function that avoids visualization issues.

**Parameters:**
- `signal_data`: ECG signal data
- `sampling_rate`: Sampling rate in Hz (default: 200)

**Returns:**
- Dictionary with analysis results

### `main()`
Main function that sets up the Streamlit interface and handles user interactions.

## Application Flow

1. **File Upload**: User uploads an EDF file
2. **File Processing**: The app reads the EDF file using MNE
3. **Holter Analysis**: For long-term recordings, the app performs Holter analysis
4. **Visualization**: The app displays the ECG signal with annotations
5. **Arrhythmia Detection**: The app detects arrhythmias in the signal
6. **Report Generation**: The app generates a medical report with findings
7. **Download Options**: User can download the report as HTML

## Custom CSS

The application includes custom CSS styling for a professional medical application appearance, including:
- Responsive layout
- Medical information cards
- Warning and normal indicators
- Section titles with consistent styling

## Usage Example

```python
# Run the Streamlit app
streamlit run ecg_streamlit_app.py
```

Then upload an EDF file through the web interface to begin analysis.

## Error Handling

The application includes robust error handling for:
- File loading issues
- Memory limitations
- Signal processing errors
- Missing or corrupt EDF headers

## Notes

- The application is designed for medical professionals and researchers
- It is not intended for clinical diagnosis without professional oversight
- Large EDF files may require significant memory resources
