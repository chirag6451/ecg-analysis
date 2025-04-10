import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import hashlib

st.set_page_config(
    page_title="ECG Dashboard Lite",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fix signal issues function
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

def plot_ecg_timeline(df, condition_spans=None, events=None):
    """
    Plot ECG timeline with events and condition spans using Plotly
    
    Args:
        df: DataFrame with time and signal columns
        condition_spans: List of dictionaries with start, end, and condition
        events: List of dictionaries with time and event
    """
    if df is None or len(df) == 0:
        print("No ECG data available for plotting")
        return None
    
    # Subsample data for performance if needed
    if len(df) > 5000:
        # Take every nth row to reduce to about 5000 points
        n = len(df) // 5000
        df = df.iloc[::n].reset_index(drop=True)
    
    # Fix signal issues
    signal = fix_signal_issues(df['signal'].values)
    time = df['time'].values
    
    # Debug info
    signal_range = np.max(signal) - np.min(signal)
    print(f"ECG Timeline Debug: Range: {signal_range:.6f}, Fixed: {True}")
    
    # Create a plotly figure
    fig = go.Figure()
    
    # Add the ECG signal
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='ECG',
        line=dict(color='rgb(200, 0, 0)', width=1.5)
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
                font=dict(color=get_condition_color(span['condition']), size=12)
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
                font=dict(color="black", size=10)
            )
    
    # Update layout
    fig.update_layout(
        title={
            "text": "ECG Timeline",
            "font": {"size": 16, "color": "black"},
            "x": 0.5
        },
        xaxis_title={
            "text": "Time (s)",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_title={
            "text": "Amplitude (mV)",
            "font": {"size": 14, "color": "black"}
        },
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        font=dict(
            size=12,
            color="rgb(0, 0, 0)"  # Black font for maximum visibility
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Explicitly set axis properties for better visibility
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(0, 0, 0, 0.5)',
        showticklabels=True,
        tickfont=dict(size=12, color="black"),
        title_standoff=15
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(0, 0, 0, 0.5)',
        showticklabels=True,
        tickfont=dict(size=12, color="black"),
        title_standoff=15
    )
    
    return fig

def plot_heart_rate_trend(data):
    """
    Plot heart rate trend
    
    Args:
        data: Dictionary with time and hr arrays
    """
    fig = go.Figure()
    
    # Subsample data for performance if needed
    if len(data['time']) > 5000:
        # Take every nth row to reduce to about 5000 points
        n = len(data['time']) // 5000
        time = data['time'][::n]
        hr = data['hr'][::n]
    else:
        time = data['time']
        hr = data['hr']
    
    fig.add_trace(go.Scatter(
        x=time,
        y=hr,
        mode='lines',
        name='Heart Rate',
        line=dict(color='rgb(192, 0, 0)', width=2.5)  # Darker red, thicker line
    ))
    
    # Add reference lines for different HR zones
    fig.add_shape(
        type="line",
        x0=min(time),
        y0=60,
        x1=max(time),
        y1=60,
        line=dict(color="rgb(153, 102, 0)", width=1.5, dash="dash"),  # Darker gold
        name="Bradycardia Threshold"
    )
    
    fig.add_shape(
        type="line",
        x0=min(time),
        y0=100,
        x1=max(time),
        y1=100,
        line=dict(color="rgb(204, 85, 0)", width=1.5, dash="dash"),  # Darker orange
        name="Tachycardia Threshold"
    )
    
    # Add colored background for different HR zones with better contrast
    fig.add_trace(go.Scatter(
        x=time + time[::-1],
        y=[100] * len(time) + [200] * len(time),
        fill='toself',
        fillcolor='rgba(204, 85, 0, 0.15)',  # Darker orange with more opacity
        line=dict(color='rgba(0,0,0,0)'),
        name='Tachycardia Zone',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=time + time[::-1],
        y=[60] * len(time) + [100] * len(time),
        fill='toself',
        fillcolor='rgba(0, 153, 51, 0.15)',  # Darker green with more opacity
        line=dict(color='rgba(0,0,0,0)'),
        name='Normal Zone',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=time + time[::-1],
        y=[0] * len(time) + [60] * len(time),
        fill='toself',
        fillcolor='rgba(153, 102, 0, 0.15)',  # Darker gold with more opacity
        line=dict(color='rgba(0,0,0,0)'),
        name='Bradycardia Zone',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title={
            "text": "Heart Rate Trend",
            "font": {"size": 16, "color": "black"},
            "x": 0.5
        },
        xaxis_title={
            "text": "Time (s)",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_title={
            "text": "Heart Rate (BPM)",
            "font": {"size": 14, "color": "black"}
        },
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent background
            bordercolor="rgba(0, 0, 0, 0.5)",     # Border for visibility
            font=dict(
                size=12,
                color="rgb(0, 0, 0)"  # Black font for maximum visibility
            )
        ),
        font=dict(
            size=12,
            color="rgb(0, 0, 0)"  # Black font for maximum visibility
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Explicitly set axis properties for better visibility
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(0, 0, 0, 0.5)',
        showticklabels=True,
        tickfont=dict(size=12, color="black"),
        title_standoff=15
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(0, 0, 0, 0.5)',
        showticklabels=True,
        tickfont=dict(size=12, color="black"),
        title_standoff=15
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
    
    # Subsample data for performance if needed
    if len(rr_intervals) > 2000:
        # Take every nth row to reduce to about 2000 points
        n = len(rr_intervals) // 2000
        rr_intervals = rr_intervals[::n]
    
    fig = go.Figure()
    
    # Plot RR(i) vs RR(i+1)
    fig.add_trace(go.Scatter(
        x=rr_intervals[:-1],
        y=rr_intervals[1:],
        mode='markers',
        marker=dict(
            color='rgba(180, 20, 20, 0.8)',  # Darker red with better opacity
            size=8,
            line=dict(
                color='rgba(40, 40, 40, 0.9)',  # Dark outline for better visibility
                width=1
            )
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
            color='rgba(0, 80, 150, 0.9)',  # Darker blue
            width=1.5,
            dash='dash'
        ),
        name='Identity Line'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            "text": "Poincaré Plot",
            "font": {"size": 16, "color": "black"},
            "x": 0.5
        },
        xaxis_title={
            "text": "RR(i) (s)",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_title={
            "text": "RR(i+1) (s)",
            "font": {"size": 14, "color": "black"}
        },
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent background
            bordercolor="rgba(0, 0, 0, 0.5)",     # Border for visibility
            font=dict(
                size=12,
                color="rgb(0, 0, 0)"  # Black font for maximum visibility
            )
        ),
        font=dict(
            size=12,
            color="rgb(0, 0, 0)"  # Black font for maximum visibility
        )
    )
    
    # Explicitly set axis properties for better visibility
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(0, 0, 0, 0.5)',
        showticklabels=True,
        tickfont=dict(size=12, color="black"),
        title_standoff=15
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.5)',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='rgba(0, 0, 0, 0.5)',
        showticklabels=True,
        tickfont=dict(size=12, color="black"),
        title_standoff=15
    )
    
    return fig

def generate_sample_data(duration=60, fs=128):
    """Generate synthetic ECG data for demo purposes"""
    # Simple sine wave with ECG-like features
    t = np.linspace(0, duration, int(duration * fs))
    
    # Base signal: sine wave
    base_signal = 0.5 * np.sin(2 * np.pi * 1.2 * t)
    
    # Add heartbeats (R peaks)
    heartbeats = np.zeros_like(t)
    heart_rate = 60  # bpm
    beat_interval = 60 / heart_rate  # seconds
    beat_locations = np.arange(0, duration, beat_interval)
    
    for loc in beat_locations:
        idx = int(loc * fs)
        if idx < len(t):
            # Create an R peak with a small QRS complex
            window = np.arange(max(0, idx-5), min(len(t), idx+5))
            r_peak = 1.0 * np.exp(-0.5 * ((t[window] - t[idx]) / 0.05)**2)
            heartbeats[window] += r_peak
    
    # Combine base and heartbeats
    signal = base_signal + heartbeats
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    signal += noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': t,
        'signal': signal
    })
    
    # Generate heart rate data
    hr_data = {
        'time': t[::fs],  # One hr point per second
        'hr': heart_rate + 5 * np.sin(2 * np.pi * 0.05 * t[::fs]) + 2 * np.random.randn(len(t[::fs]))
    }
    
    # Generate RR intervals
    rr_mean = 60 / heart_rate
    rr_intervals = rr_mean + 0.05 * np.random.randn(100)
    
    return df, hr_data, rr_intervals

def main():
    st.title("ECG Dashboard Lite")
    
    # Sidebar
    st.sidebar.header("ECG Settings")
    
    use_sample_data = st.sidebar.checkbox("Use sample data", value=True)
    
    if use_sample_data:
        # Generate sample data for demo
        duration = st.sidebar.slider("Sample Duration (seconds)", 10, 120, 60)
        df, hr_data, rr_intervals = generate_sample_data(duration=duration)
        
        # Add some condition spans for demo
        condition_spans = [
            {'start': 10, 'end': 15, 'condition': 'AF'},
            {'start': 25, 'end': 35, 'condition': 'Bradycardia'},
            {'start': 45, 'end': 55, 'condition': 'Normal'}
        ]
        
        # Add some events for demo
        events = [
            {'time': 5, 'event': 'Start'},
            {'time': 20, 'event': 'Artifact'},
            {'time': 40, 'event': 'VPB'}
        ]
    else:
        st.sidebar.info("Upload functionality not implemented in the lite version.")
        with st.sidebar.expander("Why is this disabled?"):
            st.write("""
            The full ECG Dashboard app was experiencing performance issues with large ECG files.
            This lite version focuses on providing a smooth experience with sample data.
            """)
        
        # Create empty placeholder data
        df = pd.DataFrame({'time': [], 'signal': []})
        hr_data = {'time': [], 'hr': []}
        rr_intervals = np.array([])
        condition_spans = None
        events = None
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["ECG Timeline", "Heart Rate Analysis", "Arrhythmia Analysis"])
    
    with tab1:
        st.header("ECG Signal")
        if len(df) > 0:
            ecg_fig = plot_ecg_timeline(df, condition_spans, events)
            st.plotly_chart(ecg_fig, use_container_width=True)
        else:
            st.info("No ECG data available. Please use sample data.")
    
    with tab2:
        st.header("Heart Rate Analysis")
        if len(hr_data['time']) > 0:
            # First row: Heart rate trend
            hr_fig = plot_heart_rate_trend(hr_data)
            st.plotly_chart(hr_fig, use_container_width=True)
        else:
            st.info("No heart rate data available. Please use sample data.")
    
    with tab3:
        st.header("Arrhythmia Analysis")
        if len(rr_intervals) > 1:
            # Create Poincaré plot
            poincare_fig = plot_lorenz(rr_intervals)
            st.plotly_chart(poincare_fig, use_container_width=True)
        else:
            st.info("No RR interval data available. Please use sample data.")

if __name__ == "__main__":
    main() 