# Atrial Fibrillation Detection App: User Guide

This user guide provides detailed instructions on how to use the Atrial Fibrillation Detection App, interpret results, and make the most of the advanced analysis features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Upload](#data-upload)
3. [Analysis Types](#analysis-types)
4. [Basic Analysis Features](#basic-analysis-features)
5. [Detailed Analysis Features](#detailed-analysis-features)
6. [Interpreting Results](#interpreting-results)
7. [Visualizations](#visualizations)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Getting Started

### Starting the Application

To start the AF Detection App, open a terminal and run:

```bash
streamlit run af_detection_app.py
```

This will launch the application in your default web browser. If it doesn't open automatically, navigate to the URL shown in the terminal (usually http://localhost:8501).

### Application Interface

The application interface has three main areas:

1. **Sidebar**: Contains data upload options and configuration settings
2. **Main Area**: Displays ECG signals and analysis results
3. **Information Area**: Provides educational content about AF

## Data Upload

### Supported File Formats

The application supports two types of ECG data:

#### 1. EDF (European Data Format) Files

EDF is a standard format for medical-grade ECG recordings, typically from Holter monitors.

To upload an EDF file:
1. Select "EDF (Holter)" from the file type options in the sidebar
2. Click "Browse files" to select your EDF file
3. Wait for the file to load (larger files may take more time)

#### 2. CSV/TXT Files

For simpler data formats, the app supports CSV files with ECG data.

To upload a CSV file:
1. Select "CSV/TXT" from the file type options in the sidebar
2. Click "Browse files" to select your CSV file
3. Set the appropriate sampling rate (default is 200 Hz)

CSV files should have one of these formats:
- Two columns: `time` and `signal`
- One column: just the ECG signal values
- Multiple columns: The first two will be treated as time and signal

### Segment Selection (for EDF files)

When working with EDF files (which often contain hours of data), you can select a specific segment to analyze:

1. Use the "Start time (minutes)" slider to select the starting point of your segment
2. Use the "Duration (seconds)" slider to select how long a segment to analyze (10-300 seconds)

![Segment Selection](../assets/segment_selection.png)

## Analysis Types

The app offers three analysis modes:

1. **Basic Analysis**: Shows AF probability, clinical interpretation, and HRV plots
2. **Detailed Analysis**: Provides in-depth metrics, classification results, and feature analysis
3. **Both**: Combines basic and detailed analysis (recommended for most users)

Select your preferred analysis type using the radio buttons, then click "Analyze for Atrial Fibrillation" to process the data.

## Basic Analysis Features

### AF Probability Gauge

After analysis, the app displays an AF probability gauge showing:
- Numerical probability (0-100%)
- Color-coded risk level (green: low, yellow: moderate, red: high)
- Delta indicator showing change from baseline (if available)

### Clinical Interpretation

Based on the probability, a clinical interpretation is provided:

- **High probability (≥70%)**: Indicates characteristics highly suggestive of AF
- **Moderate probability (30-70%)**: Shows some characteristics consistent with AF
- **Low probability (<30%)**: Shows minimal AF characteristics

### Heart Rate Variability Metrics

Key metrics are displayed in card format:
- **Mean Heart Rate**: Average heart rate in beats per minute (BPM)
- **Rhythm Irregularity**: Coefficient of variation of RR intervals
- **RMSSD**: Root Mean Square of Successive Differences
- **SDNN**: Standard Deviation of NN intervals
- **pNN50**: Percentage of successive RR intervals differing by more than 50ms

### HRV Visualizations

Two main visualizations are provided:

1. **RR Interval Tachogram**: Shows how the intervals between heartbeats vary over time
2. **Poincaré Plot**: Displays each RR interval against the next one, providing a visual pattern of rhythm regularity

## Detailed Analysis Features

### AF Detection Metrics

The detailed analysis expands on the basic metrics with additional information:

- More comprehensive HRV metrics display
- Probability analysis bar chart with risk thresholds
- Detailed calculation explanations

### Arrhythmia Classification Results

This section shows the distribution of different arrhythmia classes detected in the ECG:

- **Classification Table**: Shows each detected class with percentage and count
- **Classification Pie Chart**: Visual representation of class distribution
- **Interpretation**: Analysis of the classification results

### ECG Feature Analysis

The feature analysis section provides insights into the specific ECG characteristics:

- **Feature Table**: All extracted features with their values
- **Feature Bar Chart**: Visualizes key features relevant to AF detection
- **Feature Interpretation**: Explains what each feature means in the context of AF

### Educational Information

The detailed view includes an educational section about atrial fibrillation:

- **Clinical overview**: What AF is and its medical significance
- **ECG characteristics**: How AF appears on an ECG
- **Types of AF**: Different clinical categories of atrial fibrillation
- **Risk factors and management**: Information about AF causes and treatments
- **Visual comparison**: Normal sinus rhythm vs. AF patterns

## Interpreting Results

### Understanding AF Probability

The AF probability should be interpreted as follows:

- **0-30%**: Low probability of AF. The ECG likely shows normal sinus rhythm or non-AF arrhythmias.
- **30-70%**: Moderate probability of AF. Some AF characteristics are present, but not definitive.
- **70-100%**: High probability of AF. Strong evidence of atrial fibrillation in the ECG.

### Key Indicators of AF

Look for these specific indicators in the results:

1. **High Irregularity (>0.2)**: Strong indicator of the chaotic rhythm typical in AF
2. **Elevated RMSSD (>0.08s)**: Indicates high beat-to-beat variability
3. **High pNN50 (>30%)**: Shows large proportion of irregular intervals
4. **Classification as "Atrial Fibrillation"**: Direct classification by the model

### Limitations to Consider

Keep these limitations in mind when interpreting results:

- **Short segments**: Analysis of <30 seconds may not capture intermittent AF
- **Signal quality**: Poor quality recordings can affect accuracy
- **Not a clinical diagnosis**: Results should be confirmed by a healthcare professional

## Visualizations

### ECG Signal Plot

The ECG signal visualization shows:
- Raw ECG tracing in red
- Time (seconds) on the x-axis
- Signal amplitude on the y-axis
- Title showing segment information

### RR Interval Tachogram

The tachogram helps identify irregular heart rhythms:
- Regular patterns: Suggest normal rhythm
- Scattered, highly variable patterns: Suggest possible AF
- Red dashed line: Shows mean RR interval
- Green dotted lines: Show standard deviation ranges

### Poincaré Plot

This plot is particularly useful for AF detection:
- **Cigar-shaped cluster** along the identity line: Normal sinus rhythm
- **Circular, scattered pattern**: Indicative of AF
- **SD1/SD2 values**: Quantify the spread (higher SD1 suggests AF)

### Classification Pie Chart

The pie chart shows:
- Distribution of all detected arrhythmia classes
- Percentage of each class
- Color-coded segments for easy visualization

## Troubleshooting

### Common Issues

#### 1. "Error loading EDF file"
- Ensure the file is a valid EDF format
- Check if the file is corrupted or incomplete
- Try a different EDF file

#### 2. "Not enough R-peaks detected for detailed HRV analysis"
- The signal quality may be poor
- Try a different segment with clearer QRS complexes
- Increase the duration of the analyzed segment

#### 3. "Could not extract features from this ECG segment"
- The signal may have extreme noise or flatlines
- Try a different segment
- Check if the signal contains valid ECG data

#### 4. RangeError in the segment selection sliders
- This can happen with very short recordings
- The app will automatically fix the range to ensure valid values

#### 5. Warnings about missing dependencies
- The app will use fallback implementations for missing libraries
- For best results, install optional dependencies (biosppy, neurokit2)

### Performance Tips

1. For large EDF files, start with shorter segments (60 seconds)
2. If the app feels slow, try reducing the analyzed duration
3. Complex visualizations (like the Poincaré plot) may take longer to render
4. Close other browser tabs to free up memory

## Best Practices

### For Optimal Results

1. **Use clean signals**: Choose segments with minimal noise and artifacts
2. **Analyze sufficient duration**: At least 30-60 seconds for reliable AF detection
3. **Compare multiple segments**: AF can be intermittent, so check different parts of the recording
4. **Check both basic and detailed analysis**: They provide complementary information
5. **Look at the raw ECG**: Visual inspection can confirm or question algorithmic results

### Clinical Considerations

1. **This is not a medical device**: Results should be considered as decision support, not diagnosis
2. **Consult healthcare professionals**: Share results with qualified medical personnel
3. **Use in context**: Consider the patient's medical history and symptoms
4. **Follow up**: AF detection should lead to appropriate clinical follow-up
5. **Documentation**: Save or export results for inclusion in medical records if needed

## Additional Resources

For more information about atrial fibrillation and ECG analysis, consider these resources:

1. American Heart Association: [What is Atrial Fibrillation?](https://www.heart.org/en/health-topics/atrial-fibrillation)
2. Heart Rhythm Society: [Patient Resources](https://www.hrsonline.org/patient-resources)
3. UpToDate: [Patient education: Atrial fibrillation (Beyond the Basics)](https://www.uptodate.com/contents/atrial-fibrillation-beyond-the-basics)
4. PhysioNet: [ECG Databases](https://physionet.org/about/database/)

For technical details about the implementation, refer to:
- `af_detection_technical.md`: Technical documentation
- `af_detection_app_overview.md`: Application overview 