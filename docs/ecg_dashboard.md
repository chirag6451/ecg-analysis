# ECG Dashboard Documentation

## Overview

The `ecg_dashboard.py` is a visualization-focused Streamlit application for ECG data analysis. It provides a comprehensive dashboard with multiple visualization options for ECG signals, heart rate trends, and rhythm analysis. Unlike the other ECG applications, this one focuses primarily on visualization rather than medical analysis.

## Features

- **ECG Timeline Visualization**: Interactive plots of ECG signals with condition spans and events
- **Heart Rate Trend Analysis**: Visualizes heart rate changes over time with zone indicators
- **Poincaré Plot**: Visualizes heart rate variability through RR interval analysis
- **Sample Data Generation**: Can generate synthetic ECG data for demonstration purposes
- **Signal Issue Detection**: Identifies and fixes common issues with ECG signals
- **Multiple Visualization Options**: Supports both Plotly and Matplotlib for different visualization needs

## Dependencies

- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data handling and numerical operations
- **Matplotlib/Plotly**: Visualization libraries
- **SciPy**: Signal processing functions for peak detection

## Key Functions

### `get_condition_color(condition, alpha=1.0)`
Gets the appropriate color for an ECG condition.

**Parameters:**
- `condition`: Condition name (AF, Bradycardia, etc.)
- `alpha`: Alpha transparency value (default: 1.0)

**Returns:**
- Color string in RGBA format

### `fix_signal_issues(signal)`
Fixes common issues with ECG signals that can hinder visualization.

**Parameters:**
- `signal`: ECG signal array

**Returns:**
- Fixed signal array

### `plot_heart_rate_trend(data)`
Plots heart rate trend with zone indicators.

**Parameters:**
- `data`: Dictionary with time and heart_rate arrays

**Returns:**
- Plotly figure object

### `plot_lorenz(rr_intervals)`
Creates a Poincaré plot (Lorenz plot) of RR intervals for heart rate variability analysis.

**Parameters:**
- `rr_intervals`: Array of RR intervals

**Returns:**
- Plotly figure object

### `plot_ecg_timeline(df, condition_spans=None, events=None, use_plotly=True)`
Plots ECG timeline with condition spans and events.

**Parameters:**
- `df`: DataFrame with time and signal columns
- `condition_spans`: List of dictionaries with start, end, and condition
- `events`: List of dictionaries with time and event
- `use_plotly`: Whether to use Plotly for plotting (default: True)

**Returns:**
- Plotly figure object or Matplotlib figure

### `plot_ecg_timeline_matplotlib(df, condition_spans=None, events=None)`
Alternative version of the ECG timeline plot using Matplotlib instead of Plotly.

**Parameters:**
- `df`: DataFrame with time and signal columns
- `condition_spans`: List of dictionaries with start, end, and condition
- `events`: List of dictionaries with time and event

**Returns:**
- Matplotlib figure

### `generate_sample_ecg(duration=10, fs=200)`
Generates a sample ECG signal for demonstration purposes.

**Parameters:**
- `duration`: Duration in seconds (default: 10)
- `fs`: Sampling frequency in Hz (default: 200)

**Returns:**
- DataFrame with time and signal columns

### `generate_sample_heart_rate(duration_minutes=60, interval_seconds=10)`
Generates sample heart rate data for demonstration.

**Parameters:**
- `duration_minutes`: Duration in minutes (default: 60)
- `interval_seconds`: Interval between measurements in seconds (default: 10)

**Returns:**
- DataFrame with time and heart_rate columns

### `generate_sample_rr_intervals(count=300, regularity=0.9)`
Generates sample RR intervals for demonstration.

**Parameters:**
- `count`: Number of intervals to generate (default: 300)
- `regularity`: How regular the rhythm is (0-1 where 1 is perfectly regular) (default: 0.9)

**Returns:**
- Array of RR intervals in seconds

### `main()`
Main function that sets up the Streamlit interface with multiple tabs for different visualizations.

## Application Structure

The dashboard is organized into four tabs:

1. **ECG Upload**: For uploading and displaying ECG data
2. **ECG Timeline**: Shows ECG signal with annotations for conditions and events
3. **Heart Rate Trend**: Displays heart rate changes over time with zone indicators
4. **Poincaré Plot**: Visualizes heart rate variability through RR interval analysis

## Signal Visualization Features

- **Condition Spans**: Highlights regions of the ECG with specific conditions (AF, bradycardia, etc.)
- **Event Markers**: Marks specific events on the ECG timeline
- **Heart Rate Zones**: Indicates bradycardia, normal, and tachycardia zones on heart rate trends
- **Interactive Plots**: Zoom, pan, and hover capabilities for detailed exploration

## Sample Data Generation

The dashboard includes functions to generate synthetic data for demonstration:

- **Sample ECG**: Generates a realistic ECG signal with configurable parameters
- **Sample Heart Rate**: Creates heart rate data with natural variations
- **Sample RR Intervals**: Produces RR intervals with adjustable regularity

## Usage Example

```python
# Run the Streamlit app
streamlit run ecg_dashboard.py
```

Then either upload ECG data or use the sample data generation features.

## Notes

- This dashboard is focused on visualization rather than medical analysis
- It provides educational value for understanding ECG signals and heart rate patterns
- The sample data generation features are useful for demonstration and testing
- The application handles common signal issues like NaN values, Inf values, and flat lines
