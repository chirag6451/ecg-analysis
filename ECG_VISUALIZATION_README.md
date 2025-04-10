# ECG Visualization Debugging Tools

This document provides information about the new diagnostic tools added to help troubleshoot and fix ECG visualization issues, particularly "blank chart" problems.

## Overview

We've implemented several new tools to address ECG visualization problems:

1. **ECG Visualization Debugger (`ecg_viz_debug.py`)**: A standalone diagnostics app for analyzing and fixing ECG signal issues that cause blank or invisible charts.

2. **Enhanced ECG Visualization Functions**: Robust signal processing has been added to key visualization functions across the application:
   - `plot_ecg_timeline` in `ecg_dashboard.py` and `combined_ecg_app.py`
   - `plot_ecg` and `plot_ecg_with_peaks` in `enhanced_ecg_app.py`
   
3. **Signal Processing Utility Functions**: Common utility functions for fixing signal issues have been implemented, including:
   - Handling NaN values through interpolation
   - Handling Inf values by replacing with valid min/max
   - Normalizing low-amplitude signals for better visibility
   - Generating detailed signal statistics

## Using the Diagnostic Tool

To run the ECG Visualization Debugger:

```bash
streamlit run ecg_viz_debug.py
```

The diagnostic tool provides:

1. **Signal Statistics**: Detailed analysis of the signal quality, including range, NaN/Inf detection, and flatline detection.
2. **Visualization Methods Comparison**: Test different plotting methods to identify which works best.
3. **Signal Enhancement**: Apply various automatic fixes to improve visualization.
4. **Code Recommendations**: Get specific code snippets to implement in your application.
5. **Downloadable Fixed Signal**: Export a corrected version of your signal data.

## Common Issues and Solutions

### Blank or Missing ECG Charts

Several issues can cause ECG charts to appear blank or not display properly:

1. **NaN or Inf Values**: These values can't be plotted and cause visualization to fail.
   - Solution: The new signal processing functions automatically detect and handle these values.
   
2. **Very Low Amplitude Signals**: Signals with extremely small ranges (< 0.01) may appear flat.
   - Solution: Automatic normalization now scales these signals to a visible range.

3. **Edge Cases in Plotly/Matplotlib**: Some plotting libraries handle edge cases differently.
   - Solution: We've standardized the plotting approach and added robust error handling.

### How The Fix Works

The core of the fix is in the `fix_signal_issues` function, which:

1. Detects NaN values and replaces them with interpolated values or the mean.
2. Detects Inf values and replaces them with the valid min/max of the signal.
3. Automatically scales very low amplitude signals to be visible.
4. Includes detailed debug information to track what was changed.

Example implementation:

```python
def fix_signal_issues(signal):
    """Fix common issues with ECG signals that cause visualization problems."""
    if signal is None or len(signal) == 0:
        return signal
    
    # Make a copy to avoid modifying the original
    fixed_signal = signal.copy()
    
    # Handle NaN values
    if np.isnan(fixed_signal).any():
        # Get valid values mask
        valid_mask = ~np.isnan(fixed_signal)
        if np.any(valid_mask):
            valid_mean = np.mean(fixed_signal[valid_mask])
            fixed_signal[np.isnan(fixed_signal)] = valid_mean
    
    # Handle Inf values
    if np.isinf(fixed_signal).any():
        valid_mask = ~np.isinf(fixed_signal)
        if np.any(valid_mask):
            valid_min = np.min(fixed_signal[valid_mask])
            valid_max = np.max(fixed_signal[valid_mask])
            fixed_signal[np.isposinf(fixed_signal)] = valid_max
            fixed_signal[np.isneginf(fixed_signal)] = valid_min
    
    # Handle low amplitude
    signal_range = np.max(fixed_signal) - np.min(fixed_signal)
    if signal_range < 0.01 and signal_range > 0:
        fixed_signal = (fixed_signal - np.min(fixed_signal)) / signal_range
    
    return fixed_signal
```

## Best Practices

To ensure reliable ECG visualization in your application:

1. **Always validate signal data** before visualization:
   ```python
   # Check for invalid values
   contains_nan = np.isnan(signal).any()
   contains_inf = np.isinf(signal).any()
   is_flat = np.std(signal) < 1e-6
   ```

2. **Fix signal issues** before plotting:
   ```python
   fixed_signal = fix_signal_issues(signal)
   ```

3. **Use Plotly with linear interpolation** for more reliable rendering:
   ```python
   fig.add_trace(go.Scatter(
       x=time, 
       y=fixed_signal,
       mode='lines',
       line=dict(
           width=2,
           shape='linear'  # Linear (not spline) for better reliability
       )
   ))
   ```

4. **Include signal statistics in debugging logs**:
   ```python
   print(f"Signal stats: range={max-min:.6f}, mean={mean:.6f}, std={std:.6f}")
   print(f"Contains NaN: {contains_nan}, Contains Inf: {contains_inf}, Is Flat: {is_flat}")
   ```

## Modifications to Existing Code

The following files have been modified with enhanced visualization capabilities:

1. **ecg_dashboard.py**: Updated `plot_ecg_timeline` function
2. **combined_ecg_app.py**: Updated `plot_ecg_timeline` function
3. **enhanced_ecg_app.py**: Updated `plot_ecg` and `plot_ecg_with_peaks` functions
4. **Added**: `ecg_viz_debug.py` - New diagnostic tool

These changes are completely backward compatible and should not impact the existing functionality while fixing visualization issues.

## Contact

If you encounter any issues with the visualization tools or have suggestions for improvement, please contact the development team. 