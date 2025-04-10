import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO
import base64
import time

def analyze_signal_quality(signal):
    """
    Analyze the quality of the ECG signal and return detailed statistics.
    
    Args:
        signal: numpy array containing the ECG signal
        
    Returns:
        dict: Dictionary containing signal quality metrics
    """
    if signal is None or len(signal) == 0:
        return {
            'is_valid': False,
            'error': 'Empty signal'
        }
    
    # Basic statistics
    stats = {
        'is_valid': True,
        'length': len(signal),
        'min': np.min(signal) if not np.isnan(np.min(signal)) else None,
        'max': np.max(signal) if not np.isnan(np.max(signal)) else None,
        'mean': np.mean(signal) if not np.isnan(np.mean(signal)) else None,
        'median': np.median(signal) if not np.isnan(np.median(signal)) else None,
        'std': np.std(signal) if not np.isnan(np.std(signal)) else None,
        'contains_nan': np.isnan(signal).any(),
        'nan_count': np.isnan(signal).sum(),
        'contains_inf': np.isinf(signal).any(),
        'inf_count': np.isinf(signal).sum(),
        'is_flat': False
    }
    
    # Calculate additional metrics
    if stats['min'] is not None and stats['max'] is not None:
        stats['range'] = stats['max'] - stats['min']
    else:
        stats['range'] = None
    
    if stats['std'] is not None and stats['std'] < 1e-6:
        stats['is_flat'] = True
    
    # Check signal quality
    if stats['contains_nan'] or stats['contains_inf']:
        stats['quality'] = 'poor'
        stats['quality_issues'] = ['Contains NaN or Inf values']
    elif stats['is_flat']:
        stats['quality'] = 'poor'
        stats['quality_issues'] = ['Signal is flat (no variation)']
    elif stats['range'] is not None and stats['range'] < 0.01:
        stats['quality'] = 'poor'
        stats['quality_issues'] = ['Very low amplitude signal']
    else:
        stats['quality'] = 'good'
        stats['quality_issues'] = []
    
    return stats

def fix_signal_issues(signal, fix_nans=True, fix_infs=True, fix_low_amplitude=True):
    """
    Fix common issues with ECG signals that cause visualization problems.
    
    Args:
        signal: numpy array containing the ECG signal
        fix_nans: Whether to fix NaN values
        fix_infs: Whether to fix Inf values
        fix_low_amplitude: Whether to amplify low amplitude signals
        
    Returns:
        numpy array: Fixed signal
    """
    if signal is None or len(signal) == 0:
        return signal
    
    # Make a copy to avoid modifying the original
    fixed_signal = signal.copy()
    
    # Handle NaN values
    if fix_nans and np.isnan(fixed_signal).any():
        # Count NaNs
        nan_count = np.isnan(fixed_signal).sum()
        
        # If more than 50% are NaN, this signal might be too corrupted
        if nan_count > len(fixed_signal) * 0.5:
            # Return a synthetic sine wave as placeholder
            t = np.arange(len(fixed_signal))
            fixed_signal = 0.5 * np.sin(2 * np.pi * t / 100)
            print(f"WARNING: Signal had {nan_count}/{len(fixed_signal)} NaN values. Returning synthetic signal.")
        else:
            # Get valid values mask
            valid_mask = ~np.isnan(fixed_signal)
            
            if np.any(valid_mask):
                # Get statistics of valid values
                valid_mean = np.mean(fixed_signal[valid_mask])
                
                # Replace NaNs with interpolation or mean
                if nan_count <= 5:
                    # For just a few NaNs, do linear interpolation
                    for i in range(len(fixed_signal)):
                        if np.isnan(fixed_signal[i]):
                            # Find nearest valid values before and after
                            before = i - 1
                            while before >= 0 and np.isnan(fixed_signal[before]):
                                before -= 1
                            
                            after = i + 1
                            while after < len(fixed_signal) and np.isnan(fixed_signal[after]):
                                after += 1
                            
                            # Interpolate if we found valid values
                            if before >= 0 and after < len(fixed_signal):
                                fixed_signal[i] = fixed_signal[before] + (fixed_signal[after] - fixed_signal[before]) * ((i - before) / (after - before))
                            elif before >= 0:
                                fixed_signal[i] = fixed_signal[before]
                            elif after < len(fixed_signal):
                                fixed_signal[i] = fixed_signal[after]
                            else:
                                fixed_signal[i] = valid_mean
                else:
                    # For many NaNs, just use mean
                    fixed_signal[np.isnan(fixed_signal)] = valid_mean
            else:
                # All values are NaN - create a synthetic sine wave
                t = np.arange(len(fixed_signal))
                fixed_signal = 0.5 * np.sin(2 * np.pi * t / 100)
                print(f"WARNING: All values are NaN. Returning synthetic signal.")
    
    # Handle Inf values
    if fix_infs and np.isinf(fixed_signal).any():
        # Count Infs
        inf_count = np.isinf(fixed_signal).sum()
        
        # Get valid values mask
        valid_mask = ~np.isinf(fixed_signal)
        
        if np.any(valid_mask):
            # Get min and max of valid values
            valid_min = np.min(fixed_signal[valid_mask])
            valid_max = np.max(fixed_signal[valid_mask])
            
            # Replace +Inf with max and -Inf with min
            fixed_signal[np.isposinf(fixed_signal)] = valid_max
            fixed_signal[np.isneginf(fixed_signal)] = valid_min
        else:
            # All values are Inf - create a synthetic sine wave
            t = np.arange(len(fixed_signal))
            fixed_signal = 0.5 * np.sin(2 * np.pi * t / 100)
            print(f"WARNING: All values are Inf. Returning synthetic signal.")
    
    # Handle low amplitude
    if fix_low_amplitude:
        signal_range = np.max(fixed_signal) - np.min(fixed_signal)
        
        if signal_range < 0.01 and signal_range > 0:
            # Normalize and scale to a reasonable range [0, 1]
            fixed_signal = (fixed_signal - np.min(fixed_signal)) / signal_range
            print(f"INFO: Normalized very low amplitude signal. Original range: {signal_range:.6f}")
    
    return fixed_signal

def enhance_ecg_visualization(df, title="Enhanced ECG Display"):
    """
    Create an enhanced ECG visualization with better error handling and visibility.
    
    Args:
        df: DataFrame with 'time' and 'signal' columns
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Handle various error cases
    if df is None or len(df) == 0:
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No ECG data to display",
            showarrow=False,
            font=dict(color="red", size=18)
        )
        fig.update_layout(
            title="Error: No ECG Data",
            height=400
        )
        return fig
    
    # Get signal statistics before processing
    original_signal = df['signal'].values
    signal_stats = analyze_signal_quality(original_signal)
    
    # Display debug info as annotation
    debug_text = f"Length: {signal_stats['length']}<br>"
    debug_text += f"Range: {signal_stats['range']:.6f}<br>" if signal_stats['range'] is not None else ""
    debug_text += f"Contains NaN: {signal_stats['contains_nan']}<br>"
    debug_text += f"Contains Inf: {signal_stats['contains_inf']}<br>" 
    debug_text += f"Is Flat: {signal_stats['is_flat']}<br>"
    
    # Make a copy of the dataframe
    df_fixed = df.copy()
    
    # Fix common signal issues
    fixed_signal = fix_signal_issues(original_signal)
    df_fixed['signal'] = fixed_signal
    
    # Get statistics after fixing
    fixed_stats = analyze_signal_quality(fixed_signal)
    
    # Add original signal in light gray if it was modified
    if not np.array_equal(original_signal, fixed_signal):
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=original_signal,
            mode='lines',
            line=dict(width=1, color='rgba(200,200,200,0.5)'),
            name='Original Signal'
        ))
        
        debug_text += f"<br>SIGNAL FIXED:<br>"
        debug_text += f"New Range: {fixed_stats['range']:.6f}<br>" if fixed_stats['range'] is not None else ""
    
    # Add main ECG trace with more reliable rendering settings
    fig.add_trace(go.Scatter(
        x=df_fixed['time'],
        y=df_fixed['signal'],
        mode='lines',
        line=dict(
            width=2,
            color='blue',
            shape='linear'  # Linear (not spline) for better reliability
        ),
        name='ECG Signal'
    ))
    
    # Add ECG grid
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        height=400,
        hovermode="closest",
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 0, 0, 0.1)',
            dtick=0.2  # 0.2 seconds
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 0, 0, 0.1)',
            dtick=0.5  # 0.5 mV
        ),
        plot_bgcolor='white'
    )
    
    # Add debug annotation
    fig.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        text=debug_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=10),
        align="left"
    )
    
    return fig

def test_all_plot_methods(df, fs=200, title="Visualization Method Comparison"):
    """
    Test different plotting methods to identify the most reliable one.
    
    Args:
        df: DataFrame with 'time' and 'signal' columns
        fs: Sampling frequency
        title: Plot title
    """
    st.markdown(f"## {title}")
    
    # Get signal statistics
    signal = df['signal'].values
    signal_stats = analyze_signal_quality(signal)
    
    # Display signal stats
    st.json(signal_stats)
    
    # Test cases
    with st.expander("Original Data Preview", expanded=False):
        st.dataframe(df.head(100))
    
    st.subheader("1. Standard Matplotlib (Default)")
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['time'], df['signal'])
        ax.set_title(f"Matplotlib - Default")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error with Matplotlib: {str(e)}")
    
    st.subheader("2. Fixed Signal with Matplotlib")
    try:
        fixed_signal = fix_signal_issues(signal)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['time'], fixed_signal)
        ax.set_title(f"Matplotlib - Fixed Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error with Matplotlib (fixed signal): {str(e)}")
    
    st.subheader("3. Plotly (Default)")
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['signal'],
            mode='lines',
            name='ECG Signal'
        ))
        fig.update_layout(
            title="Plotly - Default",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (mV)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error with Plotly: {str(e)}")
    
    st.subheader("4. Enhanced Visualization")
    try:
        fig = enhance_ecg_visualization(df, title="Enhanced ECG Visualization")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error with Enhanced Visualization: {str(e)}")
    
    st.subheader("5. Original vs Fixed Signal Comparison")
    try:
        fixed_signal = fix_signal_issues(signal)
        
        fig = go.Figure()
        
        # Add original
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=signal,
            mode='lines',
            name='Original Signal',
            line=dict(width=1, color='gray')
        ))
        
        # Add fixed
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=fixed_signal,
            mode='lines',
            name='Fixed Signal',
            line=dict(width=2, color='blue')
        ))
        
        fig.update_layout(
            title="Original vs Fixed Signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (mV)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed comparison of stats
        fixed_stats = analyze_signal_quality(fixed_signal)
        comparison = pd.DataFrame([signal_stats, fixed_stats], index=['Original', 'Fixed'])
        st.dataframe(comparison)
    except Exception as e:
        st.error(f"Error with comparison plot: {str(e)}")
    
    st.subheader("6. Timeline Test")
    try:
        fig = go.Figure()
        
        # Add ECG signal with wider line for better visibility
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=fix_signal_issues(signal),
            mode='lines',
            name='ECG Signal',
            line=dict(
                color='rgba(0,0,0,0.8)', 
                width=2  # Wider line for better visibility
            )
        ))
        
        # Update layout with clear grid
        fig.update_layout(
            title="ECG Timeline Test",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode="closest",
            height=300,
            margin=dict(l=10, r=10, t=50, b=30),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error with timeline test: {str(e)}")

def generate_recommendations(signal_stats):
    """
    Generate specific recommendations for fixing ECG visualization based on statistics.
    
    Args:
        signal_stats: Dictionary from analyze_signal_quality()
        
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    if signal_stats['contains_nan']:
        recommendations.append("Replace NaN values with interpolation or mean values")
        code_snippet = """
# Replace NaN values with interpolation
from scipy import interpolate
valid_indices = ~np.isnan(signal)
if np.any(valid_indices):
    x = np.arange(len(signal))
    valid_x = x[valid_indices]
    valid_y = signal[valid_indices]
    f = interpolate.interp1d(valid_x, valid_y, 
                            bounds_error=False, 
                            fill_value=(valid_y[0], valid_y[-1]))
    signal = f(x)
"""
        recommendations.append(code_snippet)
    
    if signal_stats['contains_inf']:
        recommendations.append("Replace infinite values with signal min/max")
        code_snippet = """
# Replace Inf values with signal boundaries
valid_mask = ~np.isinf(signal)
if np.any(valid_mask):
    valid_min = np.min(signal[valid_mask])
    valid_max = np.max(signal[valid_mask])
    signal[np.isposinf(signal)] = valid_max
    signal[np.isneginf(signal)] = valid_min
"""
        recommendations.append(code_snippet)
    
    if signal_stats['is_flat']:
        recommendations.append("Generate synthetic signal or check data source - signal appears to be flat")
    
    if signal_stats['range'] is not None and signal_stats['range'] < 0.01:
        recommendations.append("Amplify signal for better visibility")
        code_snippet = """
# Amplify low-amplitude signal
signal_range = np.max(signal) - np.min(signal)
if signal_range < 0.01 and signal_range > 0:
    signal = (signal - np.min(signal)) / signal_range
    # Optional: Add minimal noise to prevent zero ranges
    signal += np.random.normal(0, 0.001, size=len(signal))
"""
        recommendations.append(code_snippet)
    
    # Always add these general recommendations
    recommendations.append("Use Plotly instead of Matplotlib for more reliable rendering")
    recommendations.append("Always validate and clean signal before visualization")
    recommendations.append("Log signal statistics for debugging")
    
    return recommendations

def create_ecg_debug_report(df, fs=200):
    """
    Create a comprehensive debug report for ECG visualization issues.
    
    Args:
        df: DataFrame with 'time' and 'signal' columns
        fs: Sampling frequency
    """
    # Get signal
    signal = df['signal'].values
    
    # Generate report
    st.title("ECG Visualization Debug Report")
    
    # 1. Signal statistics
    st.header("1. Signal Statistics")
    signal_stats = analyze_signal_quality(signal)
    
    # Display as JSON and highlight issues
    with st.expander("View Detailed Signal Statistics", expanded=True):
        st.json(signal_stats)
        
        # Create color-coded warning box for key issues
        if signal_stats['quality'] == 'poor':
            st.error(f"⚠️ Poor signal quality detected: {', '.join(signal_stats['quality_issues'])}")
        else:
            st.success("✅ Signal quality appears good")
    
    # 2. Visualization Tests
    st.header("2. Visualization Tests")
    test_all_plot_methods(df, fs)
    
    # 3. Recommendations
    st.header("3. Recommendations")
    recommendations = generate_recommendations(signal_stats)
    
    for i, rec in enumerate(recommendations):
        if rec.strip().startswith('#'):
            # This is a code snippet
            st.code(rec)
        else:
            # This is a text recommendation
            st.markdown(f"**{i+1}. {rec}**")
    
    # 4. Generate downloadable fixed signal
    st.header("4. Fixed Signal Data")
    fixed_signal = fix_signal_issues(signal)
    
    # Create new dataframe with fixed signal
    df_fixed = df.copy()
    df_fixed['signal'] = fixed_signal
    
    # Create downloadable CSV
    csv = df_fixed.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Fixed Signal CSV",
        data=csv,
        file_name=f"fixed_ecg_signal_{int(time.time())}.csv",
        mime="text/csv"
    )
    
    # Show preview
    with st.expander("Preview Fixed Signal Data", expanded=False):
        st.dataframe(df_fixed.head(100))
    
    # Compare original vs fixed visualization
    fig = go.Figure()
    
    # Add original
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=signal,
        mode='lines',
        name='Original Signal',
        line=dict(width=1, color='gray')
    ))
    
    # Add fixed
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=fixed_signal,
        mode='lines',
        name='Fixed Signal',
        line=dict(width=2, color='blue')
    ))
    
    fig.update_layout(
        title="Original vs Fixed Signal",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="ECG Visualization Debugger",
        page_icon="❤️",
        layout="wide"
    )
    
    st.title("ECG Visualization Debugger")
    st.markdown("""
    This tool helps identify and fix issues with ECG signal visualization.
    Upload your ECG data or use a test sample to troubleshoot blank or missing charts.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload ECG data (CSV with time and signal columns)", 
                                    type=['csv'])
    
    # Select sample data if no file uploaded
    if not uploaded_file:
        sample_type = st.selectbox(
            "Or select a test sample",
            ["Normal ECG", "Flat Line", "Very Small Amplitude", "Contains NaN/Inf"]
        )
        
        # Generate sample data
        fs = 200
        duration = 5  # seconds
        time_data = np.arange(0, duration, 1/fs)
        
        if sample_type == "Normal ECG":
            # Simple simulated ECG
            signal_data = np.zeros_like(time_data)
            for i, t in enumerate(time_data):
                cycle_phase = (t % 0.8) / 0.8  # Normalize to [0,1]
                
                # P wave
                if cycle_phase < 0.2:
                    signal_data[i] = 0.25 * np.sin(np.pi * cycle_phase / 0.2)
                # QRS complex
                elif cycle_phase < 0.35:
                    phase = (cycle_phase - 0.2) / 0.15
                    if phase < 0.2:  # Q
                        signal_data[i] = -0.4 * phase / 0.2
                    elif phase < 0.4:  # R
                        signal_data[i] = -0.4 + 2.4 * (phase - 0.2) / 0.2
                    else:  # S
                        signal_data[i] = 2.0 - 2.4 * (phase - 0.4) / 0.6
                # T wave
                elif cycle_phase < 0.6:
                    signal_data[i] = 0.35 * np.sin(np.pi * (cycle_phase - 0.35) / 0.25)
        
        elif sample_type == "Flat Line":
            signal_data = np.zeros_like(time_data)
        
        elif sample_type == "Very Small Amplitude":
            signal_data = np.zeros_like(time_data)
            for i, t in enumerate(time_data):
                cycle_phase = (t % 0.8) / 0.8  # Normalize to [0,1]
                
                # P wave
                if cycle_phase < 0.2:
                    signal_data[i] = 0.0002 * np.sin(np.pi * cycle_phase / 0.2)
                # QRS complex
                elif cycle_phase < 0.35:
                    phase = (cycle_phase - 0.2) / 0.15
                    if phase < 0.2:  # Q
                        signal_data[i] = -0.0004 * phase / 0.2
                    elif phase < 0.4:  # R
                        signal_data[i] = -0.0004 + 0.0024 * (phase - 0.2) / 0.2
                    else:  # S
                        signal_data[i] = 0.002 - 0.0024 * (phase - 0.4) / 0.6
                # T wave
                elif cycle_phase < 0.6:
                    signal_data[i] = 0.00035 * np.sin(np.pi * (cycle_phase - 0.35) / 0.25)
        
        else:  # "Contains NaN/Inf"
            # Create signal with some NaN and Inf values
            signal_data = np.zeros_like(time_data)
            for i, t in enumerate(time_data):
                cycle_phase = (t % 0.8) / 0.8
                
                if cycle_phase < 0.2:
                    signal_data[i] = 0.25 * np.sin(np.pi * cycle_phase / 0.2)
                elif cycle_phase < 0.35:
                    phase = (cycle_phase - 0.2) / 0.15
                    if phase < 0.2:
                        signal_data[i] = -0.4 * phase / 0.2
                    elif phase < 0.4:
                        signal_data[i] = -0.4 + 2.4 * (phase - 0.2) / 0.2
                    else:
                        signal_data[i] = 2.0 - 2.4 * (phase - 0.4) / 0.6
                elif cycle_phase < 0.6:
                    signal_data[i] = 0.35 * np.sin(np.pi * (cycle_phase - 0.35) / 0.25)
            
            # Insert NaN and Inf values
            for i in range(10):
                idx = np.random.randint(0, len(signal_data))
                signal_data[idx] = np.nan
            
            for i in range(5):
                idx = np.random.randint(0, len(signal_data))
                signal_data[idx] = np.inf
                
            for i in range(5):
                idx = np.random.randint(0, len(signal_data))
                signal_data[idx] = -np.inf
        
        # Create dataframe
        df = pd.DataFrame({
            'time': time_data,
            'signal': signal_data
        })
        
        st.success(f"Using sample '{sample_type}' data")
    
    else:
        # Load uploaded file
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            if 'time' not in df.columns or 'signal' not in df.columns:
                if len(df.columns) >= 2:
                    # Assume first two columns are time and signal
                    df = df.iloc[:, 0:2]
                    df.columns = ['time', 'signal']
                    st.warning("Assuming first two columns are time and signal")
                else:
                    st.error("CSV must have at least two columns for time and signal data")
                    return
            
            st.success(f"Loaded file with {len(df)} data points")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    # Ask for sampling frequency
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1, max_value=2000, value=200)
    
    # Create debug report
    create_ecg_debug_report(df, fs)

if __name__ == "__main__":
    main() 