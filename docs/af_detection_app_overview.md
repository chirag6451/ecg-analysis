# Atrial Fibrillation Detection App

## Overview

The Atrial Fibrillation Detection App is a specialized tool designed to analyze ECG signals for the detection and assessment of atrial fibrillation (AF), a common cardiac arrhythmia. This application leverages machine learning techniques and heart rate variability analysis to provide a comprehensive evaluation of ECG data.

![AF Detection App Screenshot](../assets/af_detection_app_screenshot.png)

## Key Features

- **ECG Data Analysis**: Processes ECG signals from EDF (European Data Format) files or CSV/TXT formats
- **AF Probability Assessment**: Calculates the probability of atrial fibrillation with detailed metrics
- **Multi-level Analysis**: Offers basic and detailed analysis modes
- **Heart Rate Variability Analysis**: Visualizes RR intervals and creates Poincaré plots
- **Educational Content**: Provides comprehensive information about atrial fibrillation
- **Feature Visualization**: Displays key ECG features used in the classification process
- **Interactive UI**: User-friendly interface with expandable sections and informative visuals

## Installation and Dependencies

The application requires the following Python packages:

```
streamlit
pandas
numpy
matplotlib
plotly
scipy
biosppy (optional, with fallback implementation)
neurokit2 (optional, with fallback implementation)
```

Core components required:
- `ecg_arrhythmia_classification.py`: Contains the AF classifier implementation
- `ecg_holter_analysis.py`: Provides EDF file handling capabilities

## Usage

To run the application:

```bash
streamlit run af_detection_app.py
```

### Data Requirements

The app accepts two types of inputs:
1. **EDF (Holter) files**: Standard format for medical-grade ECG recording devices
2. **CSV/TXT files**: Simple format with ECG signal values (with optional time column)

### Analysis Workflow

1. Upload your ECG data through the sidebar
2. Configure input parameters (sampling rate, segment selection)
3. View the raw ECG signal
4. Select analysis mode (Basic, Detailed, or Both)
5. Click "Analyze for Atrial Fibrillation" to process the data
6. Explore the results in the various sections

## Technical Details

### ECG Arrhythmia Classifier

The core of the application is the `ECGArrhythmiaClassifier` class, which implements:

1. **Signal preprocessing**: Cleaning, normalization, and filtering of ECG signals
2. **Feature extraction**: Calculation of time-domain, frequency-domain, and HRV features
3. **AF detection algorithm**: Rule-based system for AF probability calculation
4. **Arrhythmia classification**: Machine learning-based classification of ECG segments

#### Classification Model

The classifier utilizes a Random Forest algorithm with the following characteristics:
- 100 decision trees (n_estimators=100)
- Fixed random state for reproducibility
- StandardScaler for feature normalization
- Default initialization to prevent NotFittedError

The model can classify ECG segments into 6 classes:
- Normal
- Supraventricular Premature Beat
- Premature Ventricular Contraction
- Fusion of Ventricular and Normal Beat
- Unclassifiable Beat
- Atrial Fibrillation

### AF Detection Algorithm

The detection of atrial fibrillation is based on a weighted combination of heart rate variability metrics:

```python
probability = 0.50 * prob_from_irregularity + \
             0.30 * prob_from_rmssd + \
             0.15 * prob_from_pnn50 + \
             0.05 * hr_factor
```

Where:
- **Irregularity**: Coefficient of variation of RR intervals (SDNN/mean RR)
- **RMSSD**: Root Mean Square of Successive Differences between adjacent RR intervals
- **pNN50**: Percentage of successive RR intervals differing by more than 50ms
- **HR factor**: Contribution from heart rate (higher values in 100-175 BPM range)

#### Feature Extraction

The app extracts 15 key features from ECG signals:

1. **Time-domain statistical features**:
   - Mean amplitude
   - Standard deviation
   - Maximum and minimum values
   - Percentiles (25th, 75th)
   - Median

2. **Frequency-domain features**:
   - Total power
   - Low-frequency power (0.5-8 Hz)
   - Mid-frequency power (8-20 Hz)
   - High-frequency power (20-40 Hz)

3. **Heart rate variability features**:
   - SDNN (Standard deviation of NN intervals)
   - ASDNN (Average of standard deviations)
   - RR ratio (Max/Min RR interval)
   - Kurtosis of RR intervals

### R-Peak Detection

The app uses multiple methods to detect R-peaks in ECG signals:

1. Primary method: Hamilton detector from biosppy (with fallback)
2. Secondary method: neurokit2 ecg_peaks (with fallback)
3. Final fallback: scipy.signal.find_peaks with distance constraints

## Visualization Components

### ECG Signal Visualization

Raw ECG signals are plotted using matplotlib, displaying:
- Time series data with proper scaling
- Amplitude on the y-axis and time (seconds) on the x-axis
- Clear labeling and gridlines

### Heart Rate Variability Visualization

HRV is visualized through:

1. **RR Interval Tachogram**:
   - Plots successive RR intervals against beat number
   - Displays mean RR with standard deviation boundaries
   - Calculates derived heart rate

2. **Poincaré Plot**:
   - Plots each RR interval against the next one
   - Includes identity line for reference
   - Option for confidence ellipse with SD1/SD2 calculation
   - Helps distinguish normal from irregular rhythms

### AF Probability Visualization

AF probability is presented through:
- Gauge chart with color-coded risk levels
- Bar chart with risk thresholds
- Classification results with pie chart distributions

## Performance Considerations

- The app includes signal subsampling for handling large datasets
- Fallback implementations for optional dependencies
- Error handling for various signal qualities and formats
- Cached processing to improve interactive performance
- Graceful degradation when features can't be calculated

## References and Research Basis

The atrial fibrillation detection algorithm is based on established HRV metrics and their relationship to AF:

1. **SDNN and RMSSD**: Elevated in AF due to irregular ventricular response [1]
2. **Coefficient of Variation**: Strong predictor of AF presence [2]
3. **pNN50**: Indicates high beat-to-beat variability characteristic of AF [3]
4. **Frequency domain metrics**: Reflect disorganized atrial activity [4]

### References

[1] Task Force of the European Society of Cardiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065.

[2] Dash, S., Chon, K. H., Lu, S., & Raeder, E. A. (2009). Automatic real time detection of atrial fibrillation. Annals of biomedical engineering, 37(9), 1701-1709.

[3] Tateno, K., & Glass, L. (2001). Automatic detection of atrial fibrillation using the coefficient of variation and density histograms of RR and ΔRR intervals. Medical and Biological Engineering and Computing, 39(6), 664-671.

[4] Duverney, D., Gaspoz, J. M., Pichot, V., Roche, F., Brion, R., Antoniadis, A., & Barthélémy, J. C. (2002). High accuracy of automatic detection of atrial fibrillation using wavelet transform of heart rate intervals. Pacing and Clinical Electrophysiology, 25(4), 457-462.

## Credits and Acknowledgments

This application utilizes several open-source libraries and techniques:

- **Streamlit**: For the interactive web interface
- **Biosppy**: For ECG signal processing (with fallback implementation)
- **Neurokit2**: For advanced HRV analysis (with fallback implementation)
- **Matplotlib and Plotly**: For visualization components
- **Scipy and Numpy**: For signal processing and numerical operations
- **Pandas**: For data manipulation and management

The AF detection algorithm incorporates techniques from multiple published methods for HRV-based AF detection.

## License

This application is distributed under the [MIT License](../LICENSE). 