# ECG Analysis Applications Documentation

## Overview

This repository contains a collection of ECG (Electrocardiogram) analysis applications built with Python and Streamlit. Each application serves different purposes in ECG data visualization, analysis, and interpretation.

## Applications

### 1. [ECG Streamlit App](ecg_streamlit_app.md)

The `ecg_streamlit_app.py` is a comprehensive web application for analyzing ECG data from EDF files with a focus on arrhythmia detection and medical reporting.

**Key Features:**
- EDF file upload and processing
- Holter analysis for long-term recordings
- Arrhythmia detection including atrial fibrillation
- Medical report generation
- Interactive visualizations
- Downloadable HTML reports

### 2. [ECG Dashboard](ecg_dashboard.md)

The `ecg_dashboard.py` is a visualization-focused application providing a dashboard with multiple visualization options for ECG signals, heart rate trends, and rhythm analysis.

**Key Features:**
- ECG timeline visualization with condition spans
- Heart rate trend analysis with zone indicators
- Poincaré plot for heart rate variability analysis
- Sample data generation for demonstration
- Signal issue detection and fixing
- Multiple visualization options (Plotly and Matplotlib)

### 3. [Enhanced ECG App](enhanced_ecg_app.md)

The `enhanced_ecg_app.py` is an advanced application for comprehensive ECG analysis with a focus on atrial fibrillation detection, feature extraction, and interactive visualization.

**Key Features:**
- Advanced ECG analysis with 100+ cardiac biomarkers
- Multi-classifier approach (when available)
- Interactive visualizations with annotations
- Detailed feature extraction and display
- Analysis logging for comparison
- Data export capabilities
- Signal issue detection and fixing

### 4. [ECG EDF Analyzer](test_ecg_analyzer.md)

The `test.py` application is a specialized tool for analyzing ECG data from EDF files with a focus on data validation, visualization, and diagnostics.

**Key Features:**
- Medical-style ECG display with green lines on black background
- Data validation and repair for problematic EDF files
- Automatic detection of identical channels and data issues
- Signal normalization and scaling
- Detailed diagnostic information
- Heart rate analysis and R-peak detection
- PDF report generation

## Common Dependencies

All applications rely on the following core technologies:

- **Streamlit**: Web application framework
- **NumPy/Pandas**: Data handling and numerical operations
- **Matplotlib/Plotly**: Visualization libraries
- **NeuroKit2**: ECG signal processing
- **MNE**: EDF file reading

## Custom Modules

The applications use several custom modules:

- `ecg_holter_analysis.py`: Contains the `HolterAnalyzer` class for long-term ECG analysis
- `ecg_arrhythmia_classification.py`: Contains the `ECGArrhythmiaClassifier` class for arrhythmia detection
- `ecg_medical_analysis.py`: Contains the `ECGMedicalAnalysis` class for medical reporting
- `ecg_advanced_features.py`: Contains the `ECGFeatureExtractor` class for detailed feature extraction
- `ecg_multi_classifier.py`: Optional module with the `ECGMultiClassifier` class for multi-model classification

## Getting Started

1. Install the required dependencies:
   ```
   pip install streamlit pandas numpy matplotlib plotly neurokit2 mne scipy fpdf
   ```

2. Run any of the applications:
   ```
   streamlit run ecg_streamlit_app.py
   streamlit run ecg_dashboard.py
   streamlit run enhanced_ecg_app.py
   streamlit run test.py
   ```

3. Upload an EDF file or use the sample data generation features (when available)

## Use Cases

- **Clinical Analysis**: Use `ecg_streamlit_app.py` or `enhanced_ecg_app.py` for detailed clinical analysis
- **Research Visualization**: Use `ecg_dashboard.py` for research-focused visualization
- **Education**: Use any application for educational purposes
- **Problematic Data**: Use `test.py` for handling and diagnosing issues with ECG data

## Notes

- These applications are designed for research and educational purposes
- They are not intended for clinical diagnosis without professional oversight
- Large EDF files may require significant memory resources
- The applications include detailed explanations of metrics and analyses for educational value

# AF Detection App Documentation

This directory contains comprehensive documentation for the Atrial Fibrillation (AF) Detection App. The documentation covers various aspects of the application, from user guides to technical implementation details.

## Available Documentation

| Document | Description |
|----------|-------------|
| [AF Detection App Overview](af_detection_app_overview.md) | General overview of the application, its features, and capabilities |
| [AF Detection Technical Documentation](af_detection_technical.md) | Technical details about the AF detection algorithm and classifier implementation |
| [AF Detection User Guide](af_detection_user_guide.md) | Step-by-step guide on how to use the application and interpret results |
| [ECG Dashboard](ecg_dashboard.md) | Documentation for the ECG dashboard functionality |
| [ECG Streamlit App](ecg_streamlit_app.md) | Information about the general ECG Streamlit application |
| [Enhanced ECG App](enhanced_ecg_app.md) | Details about enhanced ECG analysis features |
| [Test ECG Analyzer](test_ecg_analyzer.md) | Information about testing ECG analysis functionalities |

## Documentation Overview

### AF Detection App Documentation

The AF Detection App documentation consists of three primary documents:

1. **App Overview**: Provides a high-level description of the application, its key features, and the technologies it uses. This is the best starting point for understanding what the app does and how it works at a conceptual level.

2. **Technical Documentation**: Contains detailed information about the implementation of the AF detection algorithm, including the feature extraction process, classifier design, and the scientific basis for AF detection. This document is ideal for developers and researchers who want to understand or extend the core functionality.

3. **User Guide**: A comprehensive manual for end-users, explaining how to use the application, interpret the results, and troubleshoot common issues. This document includes screenshots and step-by-step instructions.

### Related Documentation

The docs directory also includes information about related ECG analysis tools:

- **ECG Dashboard**: Documentation for the dashboard interface for ECG visualization and analysis
- **ECG Streamlit App**: Information about the base Streamlit application for ECG analysis
- **Enhanced ECG App**: Details about additional features in the enhanced version of the ECG application
- **Test ECG Analyzer**: Information about the testing framework for ECG analysis tools

## Getting Started

If you're new to the AF Detection App, we recommend starting with the following documents in this order:

1. [AF Detection App Overview](af_detection_app_overview.md) - To understand what the app does
2. [AF Detection User Guide](af_detection_user_guide.md) - To learn how to use the app
3. [AF Detection Technical Documentation](af_detection_technical.md) - If you're interested in the technical details

## Documentation Updates

These documents are maintained alongside the application code. When features are added or modified, the relevant documentation will be updated accordingly.

If you find any discrepancies or would like to suggest improvements to the documentation, please create an issue or pull request in the repository.

## License

All documentation is provided under the same license as the application itself. See the [LICENSE](../LICENSE) file for details.
