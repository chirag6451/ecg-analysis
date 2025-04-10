# ECG EDF Analyzer Documentation

## Overview

The `test.py` application is a specialized Streamlit tool for analyzing ECG data from EDF files with a focus on data validation, visualization, and diagnostics. It features a medical-style ECG display with green lines on black background, mimicking traditional ECG monitors used in clinical settings.

## Features

- **Medical-style ECG Display**: Green lines on black background resembling clinical ECG monitors
- **Data Validation and Repair**: Automatically detects and fixes common issues in EDF files
- **Automatic Issue Detection**: Identifies identical channels, consecutive identical values, and small data ranges
- **Signal Normalization**: Scales data to make ECG signals visible regardless of amplitude
- **Diagnostic Information**: Provides detailed information about data characteristics
- **Heart Rate Analysis**: Calculates and displays heart rate and related metrics
- **R-peak Detection**: Identifies and marks R-peaks on the ECG signal
- **PDF Report Generation**: Creates downloadable PDF reports of analysis results

## Dependencies

- **Streamlit**: Web application framework
- **NumPy/Pandas**: Data handling and numerical operations
- **Matplotlib**: Visualization with custom styling for medical displays
- **MNE**: Reading and processing EDF files
- **NeuroKit2**: ECG signal processing and feature extraction
- **FPDF**: PDF report generation

## Key Functions

### `validate_and_repair_ecg_data(data, fs=128)`
Validates ECG data and attempts to repair common issues.

**Parameters:**
- `data`: ECG data array
- `fs`: Sampling frequency in Hz (default: 128)

**Returns:**
- Repaired data array
- List of issues found

### `create_ecg_style_plot(fig, ax, x, y, title, xlabel="Time (s)", ylabel="Amplitude (mV)")`
Creates a plot that looks like a real ECG monitor.

**Parameters:**
- `fig`: Matplotlib figure object
- `ax`: Matplotlib axis object
- `x`: X-axis data (time)
- `y`: Y-axis data (signal)
- `title`: Plot title
- `xlabel`: X-axis label (default: "Time (s)")
- `ylabel`: Y-axis label (default: "Amplitude (mV)")

**Returns:**
- Styled figure and axis objects

### `main()`
Main function that sets up the Streamlit interface and handles user interactions.

## Data Validation and Repair

The application performs several validation checks on the uploaded EDF data:

1. **Identical Channels Detection**: Identifies if multiple channels contain identical data
2. **Consecutive Identical Values**: Detects segments with unchanging values
3. **Small Data Range**: Identifies signals with very small amplitude ranges
4. **NaN and Inf Detection**: Checks for NaN and Inf values in the data

For each issue detected, the application applies appropriate repairs:

1. **Channel Differentiation**: Adds synthetic variations to identical channels
2. **Value Variation**: Adds small random variations to consecutive identical values
3. **Range Normalization**: Normalizes and rescales data with small ranges
4. **NaN/Inf Replacement**: Replaces problematic values with appropriate substitutes

## Visualization Features

- **ECG Monitor Style**: Green signal lines on black background with grid lines
- **Channel Preview**: Shows preview of selected channels with diagnostic information
- **Detailed Analysis**: Provides in-depth analysis of selected channels
- **R-peak Marking**: Highlights R-peaks on the processed ECG signal
- **Heart Rate Visualization**: Displays heart rate variability over time

## Diagnostic Information

The application provides detailed diagnostic information about the ECG data:

- **Signal Statistics**: Min, max, mean, and range values
- **Data Quality Indicators**: Percentage of non-zero values, NaN counts
- **Channel Comparison**: Information about similarities between channels
- **Signal Variability**: Coefficient of variation and other variability metrics

## PDF Report Generation

The application can generate a comprehensive PDF report containing:

- **File Information**: Name, channels, sampling rate
- **Analysis Results**: Heart rate, HRV metrics
- **Timestamp**: Date and time of analysis

## Usage Example

```python
# Run the Streamlit app
streamlit run test.py
```

Then upload an EDF file through the web interface to begin analysis.

## Notes

- The application is designed to handle problematic EDF files that might not display properly in other tools
- The data validation and repair functions make it possible to visualize and analyze data that would otherwise be unusable
- The medical-style display makes it easier for healthcare professionals to interpret the ECG in a familiar format
- The application provides detailed diagnostic information to help understand issues with the data
