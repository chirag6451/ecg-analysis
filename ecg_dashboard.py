import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import random
from datetime import datetime, timedelta

def get_condition_color(condition, alpha=1.0):
    """
    Get color for ECG condition
    
    Args:
        condition: Condition name (AF, Bradycardia, etc.)
        alpha: Alpha transparency value
        
    Returns:
        Color string
    """
    condition = condition.lower()
    
    if 'af' in condition or 'atrial' in condition or 'fibrillation' in condition:
        color = f'rgba(220, 20, 60, {alpha})'  # Crimson
    elif 'brady' in condition:
        color = f'rgba(255, 165, 0, {alpha})'  # Orange
    elif 'tachy' in condition:
        color = f'rgba(255, 69, 0, {alpha})'   # Red-Orange
    elif 'normal' in condition:
        color = f'rgba(46, 139, 87, {alpha})'  # Sea Green
    elif 'vt' in condition or 'ventricular' in condition:
        color = f'rgba(139, 0, 0, {alpha})'    # Dark Red
    elif 'pause' in condition or 'arrest' in condition:
        color = f'rgba(128, 0, 128, {alpha})'  # Purple
    elif 'artifact' in condition or 'noise' in condition:
        color = f'rgba(128, 128, 128, {alpha})' # Gray
    else:
        color = f'rgba(70, 130, 180, {alpha})' # Steel Blue
        
    return color

def fix_signal_issues(signal):
    """
    Fix common issues with ECG signals that can hinder visualization
    
    Args:
        signal: ECG signal array
        
    Returns:
        Fixed signal array
    """
    if signal is None or len(signal) == 0:
        return signal
    
    # Make a copy to avoid modifying the original
    signal = np.array(signal).copy()
    
    # Handle NaN values
    nan_count = np.isnan(signal).sum()
    if nan_count > 0:
        if nan_count < 0.2 * len(signal):  # Less than 20% NaNs
            # Use linear interpolation for a few NaNs
            nan_indices = np.isnan(signal)
            non_nan_indices = ~nan_indices
            if np.any(non_nan_indices):  # Make sure there are valid values
                valid_indices = np.where(~nan_indices)[0]
                valid_values = signal[~nan_indices]
                # Create interpolator
                interp_f = lambda x: np.interp(x, valid_indices, valid_values)
                # Apply to NaN indices
                signal[nan_indices] = interp_f(np.where(nan_indices)[0])
            else:
                # All NaN, replace with zeros
                signal = np.zeros_like(signal)
        else:
            # Too many NaNs, use mean or zero
            if np.any(~np.isnan(signal)):
                signal[np.isnan(signal)] = np.nanmean(signal)
            else:
                signal = np.zeros_like(signal)
    
    # Handle Inf values
    inf_count = np.isinf(signal).sum()
    if inf_count > 0:
        # Replace +Inf with max of valid values, -Inf with min
        valid_signal = signal[~np.isinf(signal)]
        if len(valid_signal) > 0:
            max_val = np.max(valid_signal)
            min_val = np.min(valid_signal)
            signal[signal == np.inf] = max_val
            signal[signal == -np.inf] = min_val
        else:
            # All Inf, replace with zeros
            signal = np.zeros_like(signal)
    
    # If the signal range is very small (flat line), add minimal noise for visibility
    signal_range = np.max(signal) - np.min(signal)
    if signal_range < 0.01:
        # Add minimal noise (0.5% of signal mean or 0.001 if mean is zero)
        noise_amplitude = max(np.abs(np.mean(signal)) * 0.005, 0.001)
        noise = np.random.normal(0, noise_amplitude, len(signal))
        signal = signal + noise
        print(f"WARNING: Very small signal range ({signal_range:.6f}). Adding minimal noise for visibility.")
    
    return signal

def plot_heart_rate_trend(data):
    """
    Plot heart rate trend
    
    Args:
        data: Dictionary with time and heart_rate arrays
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['heart_rate'],
        mode='lines',
        name='Heart Rate',
        line=dict(color='darkred', width=2)
    ))
    
    # Add reference lines for different HR zones
    fig.add_shape(
        type="line",
        x0=min(data['time']),
        y0=60,
        x1=max(data['time']),
        y1=60,
        line=dict(color="orange", width=1.5, dash="dash"),
        name="Bradycardia Threshold"
    )
    
    fig.add_shape(
        type="line",
        x0=min(data['time']),
        y0=100,
        x1=max(data['time']),
        y1=100,
        line=dict(color="red", width=1.5, dash="dash"),
        name="Tachycardia Threshold"
    )
    
    # Add colored background for different HR zones
    fig.add_trace(go.Scatter(
        x=data['time'] + data['time'][::-1],
        y=[100] * len(data['time']) + [200] * len(data['time']),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Tachycardia Zone',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=data['time'] + data['time'][::-1],
        y=[60] * len(data['time']) + [100] * len(data['time']),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Normal Zone',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=data['time'] + data['time'][::-1],
        y=[0] * len(data['time']) + [60] * len(data['time']),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Bradycardia Zone',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Heart Rate Trend",
            font=dict(size=22, color='black')
        ),
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(size=16, color='black')),
            tickfont=dict(size=14, color='black'),
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title=dict(text="Heart Rate (BPM)", font=dict(size=16, color='black')),
            tickfont=dict(size=14, color='black'),
            gridcolor='lightgrey'
        ),
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='black')
        ),
        plot_bgcolor='white'
    )
    
    return fig

def plot_lorenz(rr_intervals):
    """
    Plot Lorenz/Poincaré plot of RR intervals
    
    Args:
        rr_intervals: Array of RR intervals
    """
    if len(rr_intervals) < 2:
        return None
    
    fig = go.Figure()
    
    # Plot RR(i) vs RR(i+1)
    fig.add_trace(go.Scatter(
        x=rr_intervals[:-1],
        y=rr_intervals[1:],
        mode='markers',
        marker=dict(
            color='darkred',
            size=8,
            line=dict(color='black', width=1)
        ),
        name='RR Intervals'
    ))
    
    # Add identity line
    min_rr = min(rr_intervals)
    max_rr = max(rr_intervals)
    
    fig.add_trace(go.Scatter(
        x=[min_rr, max_rr],
        y=[min_rr, max_rr],
        mode='lines',
        line=dict(
            color='darkblue',
            width=1.5,
            dash='dash'
        ),
        name='Identity Line'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Poincaré Plot",
            font=dict(size=22, color='black')
        ),
        xaxis=dict(
            title=dict(text="RR(i) (s)", font=dict(size=16, color='black')),
            tickfont=dict(size=14, color='black'),
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title=dict(text="RR(i+1) (s)", font=dict(size=16, color='black')),
            tickfont=dict(size=14, color='black'),
            gridcolor='lightgrey'
        ),
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(font=dict(size=12, color='black')),
        plot_bgcolor='white'
    )
    
    return fig

def plot_ecg_timeline(df, condition_spans=None, events=None, use_plotly=True):
    """
    Plot ECG timeline with events and condition spans
    
    Args:
        df: DataFrame with time and signal columns
        condition_spans: List of dictionaries with start, end, and condition
        events: List of dictionaries with time and event
        use_plotly: Whether to use plotly for plotting (defaults to True)
    """
    if df is None or len(df) == 0:
        print("No ECG data available for plotting")
        return None
    
    # Fix signal issues
    signal = fix_signal_issues(df['signal'].values)
    time = df['time'].values
    
    # Debug info
    signal_range = np.max(signal) - np.min(signal)
    print(f"ECG Timeline Debug: Range: {signal_range:.6f}, Fixed: {True}")
    
    # Subsample data for performance if needed
    if len(df) > 5000:
        # Take every nth row to reduce to about 5000 points
        n = len(df) // 5000
        time = time[::n]
        signal = signal[::n]
        print(f"Subsampled data from {len(df)} to {len(time)} points for performance")
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add the ECG signal
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='ECG',
        line=dict(color='red', width=1.5)
    ))
    
    # Add condition spans as colored backgrounds
    if condition_spans is not None:
        for span in condition_spans:
            fig.add_shape(
                type="rect",
                x0=span['start'],
                y0=np.min(signal) - 0.1 * (np.max(signal) - np.min(signal)),
                x1=span['end'],
                y1=np.max(signal) + 0.1 * (np.max(signal) - np.min(signal)),
                fillcolor=get_condition_color(span['condition'], alpha=0.2),
                line=dict(color="rgba(0,0,0,0)"),
                layer="below"
            )
            # Add text annotation
            fig.add_annotation(
                x=(span['start'] + span['end']) / 2,
                y=np.max(signal) + 0.15 * (np.max(signal) - np.min(signal)),
                text=span['condition'],
                showarrow=False,
                font=dict(color=get_condition_color(span['condition']))
            )
    
    # Add events as vertical lines
    if events is not None:
        for event in events:
            fig.add_shape(
                type="line",
                x0=event['time'],
                y0=np.min(signal) - 0.1 * (np.max(signal) - np.min(signal)),
                x1=event['time'],
                y1=np.max(signal) + 0.1 * (np.max(signal) - np.min(signal)),
                line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash")
            )
            # Add text annotation
            fig.add_annotation(
                x=event['time'],
                y=np.min(signal) - 0.2 * (np.max(signal) - np.min(signal)),
                text=event['event'],
                showarrow=False,
                font=dict(color="black")
            )
    
    # Update layout
    fig.update_layout(
        title="ECG Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30)
    )
    
    return fig

def plot_ecg_timeline_matplotlib(df, condition_spans=None, events=None):
    """
    Plot ECG timeline using matplotlib instead of plotly
    
    Args:
        df: DataFrame with time and signal columns
        condition_spans: List of dictionaries with start, end, and condition
        events: List of dictionaries with time and event
    """
    if df is None or len(df) == 0:
        print("No ECG data available for plotting")
        return None
    
    # Fix signal issues
    signal = fix_signal_issues(df['signal'].values)
    time = df['time'].values
    
    # Debug info
    signal_range = np.max(signal) - np.min(signal)
    print(f"ECG Timeline Debug (Matplotlib): Range: {signal_range:.6f}, Fixed: {True}")
    
    # Subsample data for performance if needed
    if len(df) > 10000:
        # Take every nth row to reduce to about 10000 points
        n = len(df) // 10000
        time = time[::n]
        signal = signal[::n]
        print(f"Subsampled data from {len(df)} to {len(time)} points for performance")
    
    # Create figure and plot ECG signal
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, signal, color='red', linewidth=1, label='ECG')
    
    # Set x and y limits with some padding
    y_range = np.max(signal) - np.min(signal)
    y_min = np.min(signal) - 0.1 * y_range
    y_max = np.max(signal) + 0.1 * y_range
    ax.set_ylim(y_min, y_max)
    
    # Add condition spans as colored backgrounds
    if condition_spans is not None:
        for span in condition_spans:
            color = get_condition_color(span['condition'], alpha=0.2)
            # Convert rgba string to tuple for matplotlib
            # Format: 'rgba(r, g, b, a)' -> (r/255, g/255, b/255, a)
            rgba = color.replace('rgba(', '').replace(')', '').split(',')
            color_tuple = (int(rgba[0])/255, int(rgba[1])/255, int(rgba[2])/255, float(rgba[3]))
            
            ax.axvspan(span['start'], span['end'], color=color_tuple, alpha=0.3)
            ax.text((span['start'] + span['end']) / 2, y_max, span['condition'], 
                    ha='center', va='bottom', color=color_tuple[:3])
    
    # Add events as vertical lines
    if events is not None:
        for event in events:
            ax.axvline(event['time'], color='black', linestyle='--', alpha=0.5)
            ax.text(event['time'], y_min, event['event'], 
                    ha='center', va='top', rotation=90)
    
    # Add labels and grid
    ax.set_title('ECG Timeline', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Amplitude (mV)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Improve the appearance
    fig.tight_layout()
    return fig

def generate_sample_ecg(duration=10, fs=200):
    """
    Generate a sample ECG signal for demonstration
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        
    Returns:
        DataFrame with time and signal columns
    """
    # Create time array
    time = np.arange(0, duration, 1/fs)
    
    # Initialize signal array
    signal = np.zeros_like(time)
    
    # Basic ECG-like pattern
    heart_rate = 60  # bpm
    beat_duration = 60 / heart_rate  # seconds
    
    # For each heartbeat
    for i in range(int(duration / beat_duration) + 1):
        t_center = i * beat_duration
        
        # P wave (small bump before QRS)
        p_center = t_center - 0.2
        p_width = 0.08
        p_height = 0.1
        mask = (time >= (p_center - p_width)) & (time <= (p_center + p_width))
        signal[mask] += p_height * np.sin(np.pi * (time[mask] - (p_center - p_width)) / (2 * p_width))
        
        # QRS complex (main spike)
        q_center = t_center
        q_width = 0.03
        q_height = -0.2
        mask = (time >= (q_center - q_width)) & (time <= q_center)
        signal[mask] += q_height * np.sin(np.pi * (time[mask] - (q_center - q_width)) / (2 * q_width))
        
        r_center = t_center
        r_width = 0.03
        r_height = 1.0
        mask = (time >= q_center) & (time <= (q_center + r_width))
        signal[mask] += r_height * np.sin(np.pi * (time[mask] - q_center) / (2 * r_width))
        
        s_center = t_center + r_width
        s_width = 0.04
        s_height = -0.3
        mask = (time >= s_center) & (time <= (s_center + s_width))
        signal[mask] += s_height * np.sin(np.pi * (time[mask] - s_center) / (2 * s_width))
        
        # T wave (bump after QRS)
        t_center = t_center + 0.3
        t_width = 0.1
        t_height = 0.2
        mask = (time >= (t_center - t_width)) & (time <= (t_center + t_width))
        signal[mask] += t_height * np.sin(np.pi * (time[mask] - (t_center - t_width)) / (2 * t_width))
    
    # Add some noise
    signal += np.random.normal(0, 0.02, size=len(signal))
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'signal': signal
    })
    
    return df

def generate_sample_heart_rate(duration_minutes=60, interval_seconds=10):
    """
    Generate sample heart rate data
    
    Args:
        duration_minutes: Duration in minutes
        interval_seconds: Interval between measurements in seconds
        
    Returns:
        DataFrame with time and heart_rate columns
    """
    # Create time array
    time_points = np.arange(0, duration_minutes * 60, interval_seconds)
    
    # Base heart rate with slow oscillation
    base_hr = 75 + 10 * np.sin(2 * np.pi * time_points / (60 * 30))  # 30-minute cycle
    
    # Add random variations
    heart_rate = base_hr + np.random.normal(0, 3, size=len(time_points))
    
    # Add some "events"
    # Random tachycardia - ensure valid ranges for shorter durations
    tachycardia_max = max(5, int(duration_minutes * 0.4))
    tachycardia_min = min(4, tachycardia_max - 1)
    if tachycardia_min < tachycardia_max:
        tachycardia_start = random.randint(tachycardia_min, tachycardia_max)
        tachycardia_duration = random.randint(2, min(5, duration_minutes // 2))
        tachycardia_indices = np.where(
            (time_points >= tachycardia_start * 60) & 
            (time_points < (tachycardia_start + tachycardia_duration) * 60)
        )[0]
        if len(tachycardia_indices) > 0:
            heart_rate[tachycardia_indices] += 30
    
    # Random bradycardia - ensure valid ranges for shorter durations
    brady_min = max(tachycardia_max + 1, int(duration_minutes * 0.6))
    brady_max = max(brady_min, int(duration_minutes * 0.9))
    if brady_min < brady_max:
        bradycardia_start = random.randint(brady_min, brady_max)
        bradycardia_duration = random.randint(2, min(4, duration_minutes // 2))
        bradycardia_indices = np.where(
            (time_points >= bradycardia_start * 60) & 
            (time_points < (bradycardia_start + bradycardia_duration) * 60)
        )[0]
        if len(bradycardia_indices) > 0:
            heart_rate[bradycardia_indices] -= 25
    
    # Create time strings for x-axis display
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    time_strings = [
        (start_time + timedelta(seconds=int(t))).strftime('%H:%M:%S') 
        for t in time_points
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_strings,
        'heart_rate': heart_rate
    })
    
    return df

def generate_sample_rr_intervals(count=300, regularity=0.9):
    """
    Generate sample RR intervals
    
    Args:
        count: Number of intervals to generate
        regularity: How regular the rhythm is (0-1 where 1 is perfectly regular)
        
    Returns:
        Array of RR intervals in seconds
    """
    # Base interval for 60 bpm
    base_interval = 1.0  # seconds
    
    # Generate intervals with controlled variability
    noise_scale = (1 - regularity) * 0.2
    intervals = base_interval + np.random.normal(0, noise_scale, size=count)
    
    # Add a few outlier intervals to simulate arrhythmias
    if regularity < 0.95:
        outlier_count = int(count * 0.05)
        outlier_indices = np.random.choice(count, outlier_count, replace=False)
        
        # Some shorter intervals (premature beats)
        short_indices = outlier_indices[:outlier_count//2]
        intervals[short_indices] *= 0.6
        
        # Some longer intervals (missed beats)
        long_indices = outlier_indices[outlier_count//2:]
        intervals[long_indices] *= 1.7
    
    return intervals

def main():
    st.set_page_config(page_title="ECG Dashboard", layout="wide")
    
    st.title("ECG Visualization Dashboard")
    st.markdown("""
    This dashboard provides tools for ECG signal visualization and analysis.
    Upload your EDF or CSV/TXT file to get started.
    """)
    
    # Create tabs for the different sections
    tab1, tab2, tab3, tab4 = st.tabs(["ECG Upload", "ECG Timeline", "Heart Rate Trend", "Poincaré Plot"])
    
    # Initialize session state for storing uploaded data
    if 'ecg_df' not in st.session_state:
        st.session_state.ecg_df = None
    if 'fs' not in st.session_state:
        st.session_state.fs = 200
    
    with tab1:
        st.header("Upload ECG Data")
        
        # Option to upload real data - removed sample data option
        data_source = st.radio(
            "Select data source",
            ["Upload EDF file", "Upload CSV/TXT file"]
        )
        
        if data_source == "Upload EDF file":
            # Check if required modules are available
            try:
                from ecg_holter_analysis import HolterAnalyzer
                modules_available = True
            except ImportError:
                st.error("Could not import ECG analysis modules. Make sure you're in the correct directory.")
                st.info("Missing modules. Install using: pip install pyedflib mne")
                modules_available = False
            
            if modules_available:
                uploaded_file = st.file_uploader("Upload EDF file", type=["edf"])
                
                if uploaded_file is not None:
                    # Create a temporary file to save the uploaded EDF
                    import tempfile
                    import os
                    
                    # Show loading status
                    status_placeholder = st.empty()
                    status_placeholder.info(f"Loading EDF file ({uploaded_file.size/1048576:.1f} MB). Please wait...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Initialize Holter analyzer
                    holter_analyzer = HolterAnalyzer()
                    
                    # Load the EDF file
                    if holter_analyzer.load_edf_file(tmp_path):
                        status_placeholder.success(f"EDF file loaded successfully")
                        
                        # Display file info
                        st.markdown("### Recording Information")
                        st.markdown(f"""
                        - **Duration**: {holter_analyzer.duration_hours:.2f} hours
                        - **Sampling Rate**: {holter_analyzer.fs} Hz
                        - **Start Time**: {holter_analyzer.start_time.strftime('%Y-%m-%d %H:%M:%S') if holter_analyzer.start_time else 'Unknown'}
                        """)
                        
                        # Segment selection
                        max_minutes = int(holter_analyzer.duration_hours * 60) - 1
                        segment_start = st.slider(
                            "Start time (minutes from beginning)",
                            min_value=0,
                            max_value=max_minutes,
                            value=0,
                            step=1
                        )
                        
                        segment_duration = st.slider(
                            "Segment duration (seconds)",
                            min_value=10,
                            max_value=300,
                            value=60,
                            step=10
                        )
                        
                        # Get the segment
                        df = holter_analyzer.get_segment(segment_start, segment_duration)
                        
                        if df is not None and len(df) > 0:
                            st.session_state.ecg_df = df
                            st.session_state.fs = holter_analyzer.fs
                            
                            # Calculate time string
                            if holter_analyzer.start_time:
                                from datetime import timedelta
                                time_str = (holter_analyzer.start_time + timedelta(minutes=segment_start)).strftime("%H:%M:%S")
                            else:
                                time_str = f"{segment_start // 60:02d}:{segment_start % 60:02d}:00"
                                
                            st.success(f"Loaded segment at {time_str} with {len(df)} points at {holter_analyzer.fs} Hz")
                        else:
                            st.error("Could not extract segment from EDF file")
                    else:
                        st.error("Failed to load EDF file. The file may not be in the correct format.")
                    
                    # Clean up the temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        
        elif data_source == "Upload CSV/TXT file":
            uploaded_file = st.file_uploader("Upload CSV/TXT file", type=["csv", "txt"])
            
            if uploaded_file is not None:
                try:
                    # Try to load as CSV first
                    df = pd.read_csv(uploaded_file)
                    
                    # Check if we have expected columns
                    if 'time' in df.columns and 'signal' in df.columns:
                        st.success("File loaded successfully with time and signal columns")
                    else:
                        # If not, assume the first column is time and the second is signal
                        if len(df.columns) >= 2:
                            df.columns = ['time', 'signal'] + list(df.columns[2:])
                            st.warning("Assuming first column is time and second is signal")
                        elif len(df.columns) == 1:
                            # Only one column - assume it's just the signal
                            signal = df.iloc[:, 0].values
                            df = pd.DataFrame({
                                'time': np.arange(len(signal)) / 200,  # Default 200 Hz
                                'signal': signal
                            })
                            st.warning("Only one column found - assuming it's the signal at 200 Hz")
                    
                    # Let user specify sampling frequency
                    fs = st.number_input("Sampling Frequency (Hz)", min_value=1, max_value=2000, value=200)
                    
                    st.session_state.ecg_df = df
                    st.session_state.fs = fs
                    st.success(f"Loaded file with {len(df)} data points at {fs} Hz")
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.header("ECG Signal Timeline")
        
        if st.session_state.ecg_df is not None:
            ecg_df = st.session_state.ecg_df.copy()  # Make a copy to avoid modifying the original
            
            # Create sample condition spans (optional)
            use_annotations = st.checkbox("Add sample annotations", value=False)
            condition_spans = None
            events = None
            
            if use_annotations:
                # Create sample condition spans based on the current data
                t_max = max(ecg_df['time'])
                condition_spans = [
                    {
                        'start': t_max * 0.25,
                        'end': t_max * 0.4,
                        'condition': 'Atrial Fibrillation'
                    },
                    {
                        'start': t_max * 0.6,
                        'end': t_max * 0.75,
                        'condition': 'Bradycardia'
                    }
                ]
                
                # Create sample events
                events = [
                    {
                        'time': t_max * 0.32,
                        'event': 'PVC'
                    },
                    {
                        'time': t_max * 0.68,
                        'event': 'Pause'
                    }
                ]
            
            # Use only Matplotlib for visualization
            fig = plot_ecg_timeline_matplotlib(ecg_df, condition_spans, events)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.error("Could not plot ECG timeline. Please check your data.")
            
            # Add information about the signal for debugging
            with st.expander("Signal Statistics"):
                signal = ecg_df['signal'].values
                st.write(f"Signal shape: {signal.shape}")
                st.write(f"NaN values: {np.isnan(signal).sum()}")
                st.write(f"Inf values: {np.isinf(signal).sum()}")
                
                # Handle cases where all values might be NaN or Inf
                if np.all(np.isnan(signal)) or np.all(np.isinf(signal)):
                    st.error("All signal values are NaN or Inf. Cannot calculate statistics.")
                else:
                    st.write(f"Min value: {np.nanmin(signal)}")
                    st.write(f"Max value: {np.nanmax(signal)}")
                    st.write(f"Range: {np.nanmax(signal) - np.nanmin(signal)}")
        else:
            st.info("Please upload ECG data in the 'ECG Upload' tab")
    
    with tab3:
        st.header("Heart Rate Trend Analysis")
        
        if st.session_state.ecg_df is not None:
            ecg_df = st.session_state.ecg_df
            fs = st.session_state.fs
            
            # Calculate heart rate from ECG signal
            if st.button("Calculate Heart Rate"):
                try:
                    signal = ecg_df['signal'].values
                    
                    # Basic R-peak detection (can be improved with specific libraries)
                    from scipy.signal import find_peaks
                    
                    # Simple high-pass filter to remove baseline wander
                    filtered_signal = signal - np.convolve(signal, np.ones(int(fs*0.2))/int(fs*0.2), mode='same')
                    
                    # Find R-peaks
                    r_peaks, _ = find_peaks(filtered_signal, distance=0.5*fs, height=0.5*np.std(filtered_signal))
                    
                    if len(r_peaks) > 1:
                        # Calculate RR intervals in seconds
                        rr_intervals = np.diff(r_peaks) / fs
                        
                        # Calculate heart rate in BPM
                        heart_rates = 60 / rr_intervals
                        
                        # Create time array for heart rates
                        hr_times = ecg_df['time'].iloc[r_peaks[:-1]].values
                        
                        # Create heart rate DataFrame
                        hr_df = pd.DataFrame({
                            'time': hr_times,
                            'heart_rate': heart_rates
                        })
                        
                        # Plot Heart Rate Trend
                        fig = plot_heart_rate_trend(hr_df)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Explain the zones
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("**Bradycardia Zone** (< 60 BPM)")
                            st.markdown("Slower heart rate, may be normal during rest or for athletes")
                        with col2:
                            st.markdown("**Normal Zone** (60-100 BPM)")
                            st.markdown("Typical resting heart rate range for most adults")
                        with col3:
                            st.markdown("**Tachycardia Zone** (> 100 BPM)")
                            st.markdown("Elevated heart rate, may be due to exercise, stress, or arrhythmia")
                            
                        # Also create Poincaré Plot from the same RR intervals
                        st.session_state.rr_intervals = rr_intervals
                    else:
                        st.error("Could not detect enough R-peaks for heart rate calculation")
                except Exception as e:
                    st.error(f"Error calculating heart rate: {str(e)}")
            else:
                st.info("Click 'Calculate Heart Rate' to analyze the ECG data")
        else:
            st.info("Please upload ECG data in the 'ECG Upload' tab")
    
    with tab4:
        st.header("Poincaré Plot (RR Interval Analysis)")
        
        if 'rr_intervals' in st.session_state and st.session_state.rr_intervals is not None:
            # Use calculated RR intervals from real data
            rr_intervals = st.session_state.rr_intervals
            
            # Plot Poincaré Plot
            fig = plot_lorenz(rr_intervals)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not create Poincaré plot. Insufficient RR intervals.")
                
            # Add explanation
            st.markdown("""
            **About the Poincaré Plot:**
            
            This plot visualizes the relationship between consecutive RR intervals. 
            Each point represents an RR interval plotted against the next RR interval.
            
            - A tight cluster along the identity line indicates a regular rhythm
            - Dispersed points indicate variable rhythm (could be normal or arrhythmic)
            - Specific patterns may indicate arrhythmias like atrial fibrillation
            """)
        else:
            st.info("Calculate heart rate in the 'Heart Rate Trend' tab to generate a Poincaré plot")

if __name__ == "__main__":
    main()