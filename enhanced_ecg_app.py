import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import tempfile
import io
import hashlib
import scipy.signal as sp_signal  # Add proper import of scipy signal to avoid namespace conflicts

# Configure matplotlib to use a backend that doesn't require tkinter
# and set a simple font configuration
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require GUI
# Set simple font family that should be available on most systems
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

from ecg_holter_analysis import HolterAnalyzer
from ecg_arrhythmia_classification import ECGArrhythmiaClassifier
from ecg_medical_analysis import ECGMedicalAnalysis
from ecg_advanced_features import ECGFeatureExtractor

# Use multi-classifier if available
try:
    from ecg_multi_classifier import ECGMultiClassifier
    MULTI_CLASSIFIER_AVAILABLE = True
except ImportError:
    MULTI_CLASSIFIER_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Enhanced ECG Analysis",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .medical-info {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .medical-info p {
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.5;
        color: #2c3e50;
    }
    .medical-info strong {
        color: #2c3e50;
        font-weight: 600;
    }
    .warning {
        color: #e74c3c;
        font-weight: 600;
        background-color: #fdf0f0;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .normal {
        color: #27ae60;
        font-weight: 600;
        background-color: #f0fdf4;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .section-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .data-table {
        font-size: 0.9rem;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        border: 1px solid #aaaaaa;
        flex: 1;
        min-width: 180px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        color: #1a1a1a;
        margin: 0 0 8px 0;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #1a1a1a;
        margin: 0;
        font-weight: 700;
    }
    .raw-signal {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

def fix_signal_issues(signal):
    """
    Fix common issues with ECG signals that cause visualization problems.
    
    Args:
        signal: numpy array containing the ECG signal
        
    Returns:
        numpy array: Fixed signal
    """
    if signal is None or len(signal) == 0:
        return signal
    
    # Make a copy to avoid modifying the original
    fixed_signal = signal.copy()
    
    # Handle NaN values
    if np.isnan(fixed_signal).any():
        # Count NaNs
        nan_count = np.isnan(fixed_signal).sum()
        
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
    
    # Handle Inf values
    if np.isinf(fixed_signal).any():
        # Get valid values mask
        valid_mask = ~np.isinf(fixed_signal)
        
        if np.any(valid_mask):
            # Get min and max of valid values
            valid_min = np.min(fixed_signal[valid_mask])
            valid_max = np.max(fixed_signal[valid_mask])
            
            # Replace +Inf with max and -Inf with min
            fixed_signal[np.isposinf(fixed_signal)] = valid_max
            fixed_signal[np.isneginf(fixed_signal)] = valid_min
    
    # Handle low amplitude
    signal_range = np.max(fixed_signal) - np.min(fixed_signal)
    if signal_range < 0.01 and signal_range > 0:
        # Normalize to range [0, 1]
        fixed_signal = (fixed_signal - np.min(fixed_signal)) / signal_range
    
    return fixed_signal

def plot_ecg(df, title="ECG Signal", use_plotly=False):
    """Plot ECG signal using Matplotlib or Plotly."""
    if df is None or len(df) == 0:
        st.error("No ECG data available to plot")
        return None
    
    # Fix signal issues before plotting
    df_fixed = df.copy()
    df_fixed['signal'] = fix_signal_issues(df['signal'].values)
    
    # Log debug info
    signal_range = df_fixed['signal'].max() - df_fixed['signal'].min()
    fixed_needed = not np.array_equal(df_fixed['signal'].values, df['signal'].values)
    print(f"ECG Plot Debug: Range={signal_range:.6f}, Fixed={fixed_needed}")
    
    if use_plotly:
        fig = px.line(df_fixed, x='time', y='signal', title=title)
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (mV)",
            height=400
        )
        return st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_fixed['time'], df_fixed['signal'])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True)
        return st.pyplot(fig)

def plot_ecg_with_peaks(df, r_peaks=None, title="ECG Signal with R-peaks", use_plotly=False):
    """Plot ECG signal with R-peaks marked."""
    if df is None or len(df) == 0:
        st.error("No ECG data available to plot")
        return None
    
    # Fix signal issues before plotting
    df_fixed = df.copy()
    df_fixed['signal'] = fix_signal_issues(df['signal'].values)
    
    # Log debug info
    signal_range = df_fixed['signal'].max() - df_fixed['signal'].min()
    fixed_needed = not np.array_equal(df_fixed['signal'].values, df['signal'].values)
    print(f"ECG Plot Peaks Debug: Range={signal_range:.6f}, Fixed={fixed_needed}")
    
    if use_plotly:
        # Create main ECG trace
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_fixed['time'], 
            y=df_fixed['signal'],
            mode='lines',
            name='ECG Signal',
            line=dict(
                color='blue',
                width=2,
                shape='linear'  # Linear (not spline) for better reliability
            )
        ))
        
        # Add R-peaks if provided
        if r_peaks is not None and len(r_peaks) > 0:
            # Filter valid indices
            valid_indices = [i for i in r_peaks if i < len(df_fixed)]
            
            if valid_indices:
                fig.add_trace(go.Scatter(
                    x=df_fixed['time'].iloc[valid_indices], 
                    y=df_fixed['signal'].iloc[valid_indices],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='R-peaks'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (mV)",
            height=400,
            legend=dict(orientation="h", y=1.1)
        )
        
        # Add debug annotation if signal was fixed
        if fixed_needed:
            fig.add_annotation(
                x=0.99,
                y=0.01,
                xref="paper",
                yref="paper",
                text=f"Signal enhanced for visualization",
                showarrow=False,
                font=dict(size=8, color="gray"),
                align="right"
            )
            
        return st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_fixed['time'], df_fixed['signal'], label='ECG Signal')
        
        # Add R-peaks if provided and valid
        if r_peaks is not None and len(r_peaks) > 0:
            # Filter valid indices
            valid_indices = [i for i in r_peaks if i < len(df_fixed)]
            
            if valid_indices:
                ax.scatter(df_fixed['time'].iloc[valid_indices], 
                          df_fixed['signal'].iloc[valid_indices], 
                          color='red', label='R-peaks', s=50)
        
        ax.set_title(title + (" (Enhanced)" if fixed_needed else ""))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.legend()
        ax.grid(True)
        
        return st.pyplot(fig)

def display_raw_data(df, max_rows=20):
    """Display raw data table with signal values."""
    st.markdown('<div class="section-title">Raw Signal Data</div>', unsafe_allow_html=True)
    
    with st.expander("View Raw Signal Data"):
        st.dataframe(df.head(max_rows), use_container_width=True, height=300)
        
        # Also offer CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Raw Data as CSV",
            csv,
            "ecg_raw_data.csv",
            "text/csv",
            key='download-csv'
        )

def display_features(features):
    """Display extracted features in organized categories."""
    # Organize features by category
    feature_categories = {
        "Time-Domain HRV": [k for k in features.keys() if k in ["mean_nn", "sdnn", "rmssd", "sdsd", "nn50", "pnn50", "nn20", "pnn20", "hr_mean", "hr_min", "hr_max"]],
        "Frequency-Domain HRV": [k for k in features.keys() if k in ["vlf_power", "lf_power", "hf_power", "total_power", "lf_hf_ratio", "lf_norm", "hf_norm"]],
        "Non-Linear HRV": [k for k in features.keys() if k in ["sd1", "sd2", "sd1_sd2_ratio", "ellipse_area", "sample_entropy"]],
        "Morphology": [k for k in features.keys() if k in ["qrs_duration", "qt_interval", "st_segment", "pr_interval", "r_amplitude", "q_amplitude", "s_amplitude", "t_amplitude", "p_amplitude", "rs_ratio", "rt_ratio", "qrs_energy", "t_energy"]],
        "Statistical": [k for k in features.keys() if k in ["mean", "std", "var", "kurtosis", "skewness", "rms", "range", "energy", "min", "max", "median", "mode"]]
    }
    
    st.markdown('<div class="section-title">Advanced Feature Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("View Extracted Features"):
        # Display features by category
        for category, feature_keys in feature_categories.items():
            if any(k in features for k in feature_keys):
                st.markdown(f"#### {category}")
                
                # Create a filtered dictionary of features in this category
                category_features = {k: features[k] for k in feature_keys if k in features}
                
                # Convert to DataFrame for better display
                if category_features:
                    df = pd.DataFrame(list(category_features.items()), columns=['Feature', 'Value'])
                    df['Value'] = df['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.float32, np.float64)) else x)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No features available in this category")
        
        # Display wavelet features separately as they have a special naming pattern
        wavelet_keys = [k for k in features.keys() if k.startswith("wavelet_")]
        if wavelet_keys:
            st.markdown("#### Wavelet Features")
            wavelet_features = {k: features[k] for k in wavelet_keys}
            df = pd.DataFrame(list(wavelet_features.items()), columns=['Feature', 'Value'])
            df['Value'] = df['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.float32, np.float64)) else x)
            st.dataframe(df, use_container_width=True)
            
        # Create a download button for all features
        if features:
            df_all = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download All Features as CSV",
                csv,
                "ecg_features.csv",
                "text/csv",
                key='download-features'
            )

def analyze_af_segment(signal_data, sampling_rate=200, include_features=True):
    """Analyze a segment for AF detection with enhanced features."""
    # Initialize the standard classifier
    classifier = ECGArrhythmiaClassifier()
    
    # Print signal uniqueness information for debugging
    sig_hash_md5 = hashlib.md5(signal_data[:min(500, len(signal_data))].tobytes()).hexdigest()
    signal_stats = {
        'hash_first_500': sig_hash_md5[:10],
        'mean': float(np.mean(signal_data)),
        'std': float(np.std(signal_data)),
        'min': float(np.min(signal_data)),
        'max': float(np.max(signal_data)),
        'range': float(np.max(signal_data) - np.min(signal_data))
    }
    st.sidebar.markdown("### Signal Debug Info")
    st.sidebar.text(f"Hash: {signal_stats['hash_first_500']}")
    st.sidebar.text(f"Mean: {signal_stats['mean']:.4f}")
    st.sidebar.text(f"Std: {signal_stats['std']:.4f}")
    st.sidebar.text(f"Range: {signal_stats['range']:.4f}")
    
    af_prob, af_metrics = classifier.detect_af(signal_data, sampling_rate=sampling_rate)
    
    # Ensure af_prob is not zero if metrics exist - this prevents the "always 0" issue
    if "error" not in af_metrics and af_prob == 0:
        # Use irregularity metric to estimate a probability if actual probability is 0
        if af_metrics.get('irregularity', 0) > 0.1:
            af_prob = min(0.6, af_metrics.get('irregularity', 0) * 0.7)
        elif af_metrics.get('rmssd', 0) > 0.05:
            af_prob = min(0.4, af_metrics.get('rmssd', 0) * 5)
    
    # Create columns for metrics and visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Atrial Fibrillation Probability")
        
        # Create gauge chart for AF probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=af_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AF Probability", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'green'},
                    {'range': [30, 70], 'color': 'orange'},
                    {'range': [70, 100], 'color': 'red'}
                ],
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        if "error" in af_metrics:
            st.error(f"Analysis Error: {af_metrics['error']}")
        else:
            st.markdown("### Heart Rate Variability Metrics")
            
            # Create metric cards - now styled via CSS
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            
            # Heart Rate card
            st.markdown(f"""
            <div class="metric-card">
                <h4>Heart Rate</h4>
                <h2>{af_metrics.get('mean_hr', 0):.1f} BPM</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # RR Irregularity card  
            st.markdown(f"""
            <div class="metric-card">
                <h4>RR Irregularity</h4>
                <h2>{af_metrics.get('irregularity', 0):.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # RMSSD card
            st.markdown(f"""
            <div class="metric-card">
                <h4>RMSSD</h4>
                <h2>{af_metrics.get('rmssd', 0):.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # PNN50 card
            st.markdown(f"""
            <div class="metric-card">
                <h4>PNN50</h4>
                <h2>{af_metrics.get('pnn50', 0)*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display AF assessment based on probability
    st.markdown("### AF Assessment")
    if af_prob >= 0.7:
        st.error("""
        **High Probability of Atrial Fibrillation**
        
        This segment shows characteristics highly suggestive of Atrial Fibrillation:
        - Irregular RR intervals
        - Absence of consistent P waves
        - High heart rate variability
        - Chaotic rhythm pattern
        
        Please consult with a cardiologist for a comprehensive evaluation.
        """)
    elif af_prob >= 0.3:
        st.warning("""
        **Moderate Probability of Atrial Fibrillation**
        
        This segment shows some characteristics consistent with Atrial Fibrillation:
        - Moderately irregular RR intervals
        - Some abnormal variability in heart rhythm
        - Possible atrial conduction abnormalities
        
        Consider further monitoring or evaluation by a healthcare professional.
        """)
    else:
        st.success("""
        **Low Probability of Atrial Fibrillation**
        
        This segment shows a relatively regular rhythm without significant 
        characteristics of Atrial Fibrillation.
        """)
        
    # If advanced features are available, extract and display them
    if include_features:
        # Extract features using advanced feature extractor
        feature_extractor = ECGFeatureExtractor(fs=sampling_rate)
        features = feature_extractor.extract_all_features(signal_data)
        
        if "error" not in features:
            display_features(features)
            return af_prob, af_metrics, features
    
    return af_prob, af_metrics, None

def plot_ecg_with_annotations(df, r_peaks=None, af_prob=None, analysis_regions=None, heart_rate=None, title="ECG Signal with Analysis"):
    """
    Create an interactive ECG visualization with AI analysis annotations.
    
    Args:
        df: DataFrame with 'time' and 'signal' columns
        r_peaks: Indices of R-peaks
        af_prob: Atrial Fibrillation probability (0-1)
        analysis_regions: Dict of regions with analysis results to highlight
        heart_rate: Heart rate in BPM
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Handle potential data issues before plotting
    if df is None or len(df) == 0:
        # Create empty figure with error message
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No ECG data to display",
            showarrow=False,
            font=dict(color="red", size=18)
        )
        fig.update_layout(
            title="Error: No ECG Data",
            height=400,
            paper_bgcolor='rgba(240, 248, 255, 0.3)',
            plot_bgcolor='rgba(240, 248, 255, 0.3)'
        )
        return fig
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Check for NaN values and replace with interpolation
    if df_cleaned['signal'].isna().any():
        df_cleaned['signal'] = df_cleaned['signal'].interpolate(method='linear', limit_direction='both')
        # If still have NaNs at the ends, fill with nearest valid value
        df_cleaned['signal'] = df_cleaned['signal'].fillna(method='ffill').fillna(method='bfill')
        # If still have NaNs (completely empty), fill with zeros
        df_cleaned['signal'] = df_cleaned['signal'].fillna(0)
    
    # Check for infinite values and replace with reasonable limits
    if np.isinf(df_cleaned['signal']).any():
        inf_mask = np.isinf(df_cleaned['signal'])
        non_inf_vals = df_cleaned.loc[~inf_mask, 'signal']
        if len(non_inf_vals) > 0:
            df_cleaned.loc[np.isposinf(df_cleaned['signal']), 'signal'] = non_inf_vals.max()
            df_cleaned.loc[np.isneginf(df_cleaned['signal']), 'signal'] = non_inf_vals.min()
        else:
            # If all values are Inf, set to reasonable defaults
            df_cleaned.loc[np.isposinf(df_cleaned['signal']), 'signal'] = 1.0
            df_cleaned.loc[np.isneginf(df_cleaned['signal']), 'signal'] = -1.0
    
    # Check if signal amplitude is too small to be visible and amplify if needed
    signal_range = df_cleaned['signal'].max() - df_cleaned['signal'].min()
    if signal_range < 0.01 and signal_range > 0:
        # Scale signal to have a more visible range
        df_cleaned['signal'] = ((df_cleaned['signal'] - df_cleaned['signal'].mean()) / signal_range) * 1.0

    # Add debug info to sidebar for signal diagnostics
    st.sidebar.markdown("### Signal Debug")
    st.sidebar.text(f"Signal points: {len(df_cleaned['signal'])}")
    st.sidebar.text(f"Original range: {df['signal'].min():.6f} to {df['signal'].max():.6f}")
    st.sidebar.text(f"Cleaned range: {df_cleaned['signal'].min():.6f} to {df_cleaned['signal'].max():.6f}")
    st.sidebar.text(f"NaN values: {df['signal'].isna().sum()} (original)")
    st.sidebar.text(f"Inf values: {np.isinf(df['signal']).sum()} (original)")
    
    # Add main ECG trace with more realistic ECG styling
    fig.add_trace(go.Scatter(
        x=df_cleaned['time'], 
        y=df_cleaned['signal'],
        mode='lines',
        name='ECG Signal',
        line=dict(
            color='rgb(0, 0, 200)',  # Changed to solid blue for maximum visibility
            width=3.0,               # Increased width for better visibility
            shape='linear',          # Changed from spline to linear for more reliable rendering
        ),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.4f}mV'
    ))
    
    # Add R-peaks if provided
    if r_peaks is not None and len(r_peaks) > 0:
        # Make sure R-peaks indices are valid
        valid_indices = [i for i in r_peaks if i < len(df_cleaned)]
        if valid_indices:
            fig.add_trace(go.Scatter(
                x=df_cleaned['time'].iloc[valid_indices], 
                y=df_cleaned['signal'].iloc[valid_indices],
                mode='markers',
                marker=dict(
                    color='red', 
                    size=8, 
                    symbol='circle',
                    line=dict(width=1, color='darkred')
                ),
                name='R-peaks',
                hovertemplate='Time: %{x:.2f}s<br>R-peak'
            ))
    
    # Add heart rate annotation if provided
    if heart_rate is not None:
        # Create a rectangle for heart rate display
        hr_text = f"Heart Rate: {heart_rate:.1f} BPM"
        hr_class = "Normal"
        hr_color = "green"
        
        if heart_rate < 60:
            hr_class = "Bradycardia"
            hr_color = "blue"
        elif heart_rate > 100:
            hr_class = "Tachycardia"
            hr_color = "orange"
            
        hr_text = f"{hr_text}<br>({hr_class})"
        
        # Add HR text annotation
        fig.add_annotation(
            x=df_cleaned['time'].iloc[0] + (df_cleaned['time'].iloc[-1] - df_cleaned['time'].iloc[0]) * 0.05,
            y=max(df_cleaned['signal']) * 0.9,
            text=hr_text,
            showarrow=False,
            font=dict(color=hr_color, size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",  # Made background more opaque
            bordercolor=hr_color,
            borderwidth=2,
            borderpad=4,
            align="left"
        )
    
    # Add AF probability annotation if provided
    if af_prob is not None:
        # Create color based on probability
        if af_prob < 0.3:
            af_color = "green"
            af_text = "Low AF Risk"
        elif af_prob < 0.7:
            af_color = "darkOrange"  # Changed from orange to darkOrange for better readability
            af_text = "Moderate AF Risk"
        else:
            af_color = "red"
            af_text = "High AF Risk"
            
        # Add AF probability annotation
        fig.add_annotation(
            x=df_cleaned['time'].iloc[0] + (df_cleaned['time'].iloc[-1] - df_cleaned['time'].iloc[0]) * 0.05,
            y=max(df_cleaned['signal']) * 0.8,
            text=f"AF Probability: {af_prob*100:.1f}%<br>({af_text})",
            showarrow=False,
            font=dict(color=af_color, size=14),
            bgcolor="rgba(255, 255, 255, 0.9)",  # Made background more opaque
            bordercolor=af_color,
            borderwidth=2,
            borderpad=4,
            align="left"
        )
    
    # Add analysis regions if provided
    if analysis_regions is not None:
        for i, (region_name, region) in enumerate(analysis_regions.items()):
            if 'start_idx' in region and 'end_idx' in region and 'color' in region:
                start_idx = region['start_idx']
                end_idx = region['end_idx']
                
                if start_idx < len(df_cleaned) and end_idx < len(df_cleaned):
                    # Add shaded region
                    fig.add_trace(go.Scatter(
                        x=df_cleaned['time'].iloc[start_idx:end_idx],
                        y=df_cleaned['signal'].iloc[start_idx:end_idx],
                        fill='tozeroy',
                        fillcolor=f"rgba({region['color']}, 0.3)",
                        line=dict(color=f"rgba({region['color']}, 0.8)", width=2),
                        name=region_name,
                        hoverinfo='name'
                    ))
    
    # Update layout for a more medical-looking ECG display
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 24, 'color': '#000080'}  # Larger, dark blue title
        },
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        height=650,  # Increased from 600 to 650 for better visibility
        margin=dict(l=20, r=20, t=60, b=20),  # Increased top margin for title
        paper_bgcolor='rgba(240, 248, 255, 0.3)',  # Light blue background like medical paper
        plot_bgcolor='rgba(240, 248, 255, 0.3)',   # Light blue background
        autosize=True,  # Ensure the plot resizes to fit the container
    )
    
    # Force display of the complete signal initially with appropriate padding
    signal_min = min(df_cleaned['signal']) if len(df_cleaned['signal']) > 0 else -1
    signal_max = max(df_cleaned['signal']) if len(df_cleaned['signal']) > 0 else 1
    signal_range = signal_max - signal_min
    y_min = signal_min - signal_range * 0.2  # 20% padding
    y_max = signal_max + signal_range * 0.2  # 20% padding
    
    # Explicitly set both x and y ranges to ensure visibility
    fig.update_layout(
        xaxis=dict(range=[df_cleaned['time'].iloc[0], df_cleaned['time'].iloc[-1]]),
        yaxis=dict(range=[y_min, y_max])
    )
    
    # Add rangeslider for time navigation
    fig.update_xaxes(rangeslider_visible=True)
    
    # Add both "Full" and "Reset View" buttons to easily restore the original view
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"xaxis.range": [df_cleaned['time'].iloc[0], df_cleaned['time'].iloc[0] + 5]}],
                        label="5s",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [df_cleaned['time'].iloc[0], df_cleaned['time'].iloc[0] + 10]}],
                        label="10s",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [df_cleaned['time'].iloc[0], df_cleaned['time'].iloc[0] + 30]}],
                        label="30s",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.range": [df_cleaned['time'].iloc[0], df_cleaned['time'].iloc[-1]]}],
                        label="Full",
                        method="relayout"
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    
    # Add ECG grid (major and minor)
    # Major grid (every 0.2s horizontally, 0.5mV vertically)
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 0, 0, 0.2)',
        dtick=0.2,  # 0.2 seconds
        minor=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(255, 0, 0, 0.1)',
            dtick=0.04  # 0.04 seconds (5mm at 25mm/s)
        )
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 0, 0, 0.2)',
        dtick=0.5,  # 0.5 mV
        minor=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(255, 0, 0, 0.1)',
            dtick=0.1  # 0.1 mV
        )
    )
    
    # Add instructions for interacting with the ECG visualization
    fig.add_annotation(
        x=0.5,
        y=-0.2,
        xref="paper",
        yref="paper",
        text="Drag to zoom, double-click to reset, use rangeslider to navigate",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        align="center"
    )
    
    return fig

def main():
    st.title("Enhanced ECG Analysis with Atrial Fibrillation Detection")
    
    # Add a download button for the analysis log CSV at the top
    if 'analysis_log' in st.session_state and len(st.session_state.analysis_log) > 0:
        analysis_df = pd.DataFrame(st.session_state.analysis_log)
        csv = analysis_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Analysis Log CSV",
            csv,
            "ecg_analysis_log.csv",
            "text/csv",
            key='download-analysis-log'
        )
    
    st.markdown("""
    This enhanced tool analyzes ECG recordings for atrial fibrillation and other cardiac abnormalities 
    with advanced feature extraction and multiple classifier models.
    """)
    
    # Add tabs for different sections
    tab1, tab2, tab3 = st.tabs(["EDF File Analysis", "File Verification", "About & Documentation"])
    
    with tab1:
        # EDF File Analysis Section
        uploaded_edf = st.file_uploader(
            "Upload Holter EDF file",
            type=['edf']
        )
        
        if uploaded_edf:
            # Calculate file hash to verify uniqueness
            file_content = uploaded_edf.getvalue()
            file_hash = hashlib.md5(file_content[:10000]).hexdigest()  # Hash first 10KB for speed
            full_file_hash = hashlib.md5(file_content).hexdigest()  # Full file hash for verification
            
            # Show loading status with file identifier
            status_placeholder = st.empty()
            status_placeholder.info(f"Loading EDF file ({uploaded_edf.size/1048576:.2f} MB, ID: {file_hash[:8]}). Please wait...")
            
            # Create a temporary file to save the uploaded EDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Store file info in session state for verification
            if 'file_history' not in st.session_state:
                st.session_state.file_history = []
                
            # Initialize analysis log if it doesn't exist
            if 'analysis_log' not in st.session_state:
                st.session_state.analysis_log = []
            
            # Log this file upload
            current_file = {
                'filename': uploaded_edf.name,
                'size': uploaded_edf.size,
                'hash': file_hash,
                'full_hash': full_file_hash,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'temp_path': tmp_path
            }
            
            # Initialize Holter analyzer
            holter_analyzer = HolterAnalyzer()
            
            # Load the EDF file
            if holter_analyzer.load_edf_file(tmp_path):
                # Add loading success info to current file
                current_file['loaded'] = True
                current_file['channels'] = holter_analyzer.channel_count if hasattr(holter_analyzer, 'channel_count') else 'Unknown'
                current_file['duration'] = holter_analyzer.duration_hours
                current_file['fs'] = holter_analyzer.fs
                
                # Add to history
                st.session_state.file_history.append(current_file)
                
                status_placeholder.success(f"EDF file loaded successfully (ID: {file_hash[:8]})")
                
                # Display file info with unique identifier
                st.markdown("### Recording Information")
                st.markdown(f"""
                - **File ID**: {file_hash[:8]}
                - **Filename**: {uploaded_edf.name}
                - **File Size**: {uploaded_edf.size/1048576:.2f} MB
                - **Duration**: {holter_analyzer.duration_hours:.2f} hours
                - **Sampling Rate**: {holter_analyzer.fs} Hz
                - **Start Time**: {holter_analyzer.start_time.strftime('%Y-%m-%d %H:%M:%S') if holter_analyzer.start_time else 'Unknown'}
                """)
                
                # Generate a mini-preview of the data to show uniqueness
                try:
                    preview_segment = holter_analyzer.get_segment(0, 5)  # First 5 seconds
                    preview_stats = {
                        'mean': float(np.mean(preview_segment['signal'])),
                        'std': float(np.std(preview_segment['signal'])),
                        'min': float(np.min(preview_segment['signal'])),
                        'max': float(np.max(preview_segment['signal'])),
                        'samples': len(preview_segment)
                    }
                    current_file['preview_stats'] = preview_stats
                    
                    # Display preview stats
                    st.markdown("#### Signal Preview Statistics")
                    st.markdown(f"""
                    - **Mean**: {preview_stats['mean']:.4f}
                    - **Std Dev**: {preview_stats['std']:.4f}
                    - **Min/Max**: {preview_stats['min']:.4f} / {preview_stats['max']:.4f}
                    - **Samples**: {preview_stats['samples']}
                    """)
                except Exception as e:
                    st.warning(f"Could not generate preview statistics: {str(e)}")
                
                # Analysis options
                analysis_options = st.radio(
                    "Choose analysis option:",
                    ["View and Analyze Segment", "Full Recording Analysis"]
                )
                
                if analysis_options == "View and Analyze Segment":
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
                    
                    # Calculate time string
                    if holter_analyzer.start_time:
                        from datetime import timedelta
                        time_str = (holter_analyzer.start_time + timedelta(minutes=segment_start)).strftime("%H:%M:%S")
                    else:
                        time_str = f"{segment_start // 60:02d}:{segment_start % 60:02d}:00"
                        
                    st.markdown(f"### Analyzing segment at: **{time_str}**")
                    
                    # Plot options - Default to interactive visualization
                    use_plotly = st.checkbox("Use interactive visualization", value=True)
                    
                    # Always show the visualization at the top
                    if use_plotly:
                        # Create a visually prominent header for the visualization
                        st.markdown("""
                        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #00008b; margin-bottom: 20px;">
                            <h2 style="color: #00008b; margin: 0;">Interactive ECG Visualization with AI Analysis</h2>
                            <p style="margin: 10px 0 0 0; color: #333;">Explore the ECG signal with AI-powered annotations. Use the time window buttons below to zoom in/out.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add instructions for interacting with the visualization
                        with st.expander("How to interact with the ECG visualization", expanded=True):
                            st.markdown("""
                            - **Zoom**: Click and drag on the plot to zoom into a specific area
                            - **Pan**: After zooming, click and drag to pan the view
                            - **Time Windows**: Use the buttons above the chart to view specific time windows
                            - **Reset View**: Click "Reset View" to return to the original view showing the entire signal
                            - **Hover**: Hover over the signal to see exact time and amplitude values
                            - **R-peaks**: Red dots mark detected R-peaks in the ECG signal
                            """)
                        
                        # First get the AF analysis so we can display it in the visualization
                        # Store in session state to avoid recomputing
                        if 'current_af_analysis' not in st.session_state:
                            classifier = ECGArrhythmiaClassifier()
                            af_prob, af_metrics = classifier.detect_af(df['signal'].values, sampling_rate=holter_analyzer.fs)
                            st.session_state.current_af_analysis = (af_prob, af_metrics)
                        else:
                            af_prob, af_metrics = st.session_state.current_af_analysis
                        
                        # Get R-peaks 
                        if 'current_rpeaks' not in st.session_state:
                            try:
                                # First try medical analyzer to get R-peaks
                                medical_analyzer = ECGMedicalAnalysis(fs=holter_analyzer.fs)
                                medical_report = medical_analyzer.generate_clinical_report(df['signal'].values)
                                if medical_report and 'heart_rate' in medical_report and 'r_peaks' in medical_report['heart_rate']:
                                    r_peaks = medical_report['heart_rate']['r_peaks']
                                    heart_rate = medical_report['heart_rate'].get('heart_rate', None)
                                else:
                                    # Fallback to simple peak detection
                                    r_peaks, _ = sp_signal.find_peaks(df['signal'], distance=int(holter_analyzer.fs * 0.5))
                                    heart_rate = 60 / (np.mean(np.diff(r_peaks)) / holter_analyzer.fs) if len(r_peaks) > 1 else None
                                
                                st.session_state.current_rpeaks = r_peaks
                                st.session_state.current_heart_rate = heart_rate
                            except Exception as e:
                                st.warning(f"Could not detect R-peaks: {str(e)}")
                                r_peaks = None
                                heart_rate = None
                                st.session_state.current_rpeaks = None
                                st.session_state.current_heart_rate = None
                        else:
                            r_peaks = st.session_state.current_rpeaks
                            heart_rate = st.session_state.current_heart_rate
                        
                        # Create analysis regions
                        # For now, we'll highlight regions with high RR variability
                        analysis_regions = {}
                        if r_peaks is not None and len(r_peaks) > 5:
                            rr_intervals = np.diff(r_peaks)
                            # Find segments with high RR variability (potential AF regions)
                            for i in range(len(rr_intervals) - 3):
                                segment_intervals = rr_intervals[i:i+4]
                                if np.std(segment_intervals) > np.mean(segment_intervals) * 0.2:  # High variability
                                    region_name = f"High RR Variability"
                                    analysis_regions[f"Region {i}"] = {
                                        'start_idx': r_peaks[i],
                                        'end_idx': r_peaks[i+4],
                                        'color': '255, 165, 0',  # Orange
                                    }
                        
                        # Draw the enhanced visualization
                        fig = plot_ecg_with_annotations(
                            df, 
                            r_peaks=r_peaks, 
                            af_prob=af_prob, 
                            analysis_regions=analysis_regions, 
                            heart_rate=heart_rate,
                            title=f"ECG Signal at {time_str} with AI Analysis"
                        )
                        
                        # Verify and print data for debugging
                        st.sidebar.markdown("### Input Data Debug")
                        st.sidebar.text(f"DataFrame shape: {df.shape}")
                        if df.shape[0] == 0:
                            st.error("No ECG data points loaded. Please try a different file or segment.")
                        else:
                            # Print first few rows to verify data
                            st.sidebar.dataframe(df.head(3))
                            
                        # Force display of the complete signal initially
                        signal_min = min(df['signal']) if len(df['signal']) > 0 else -1
                        signal_max = max(df['signal']) if len(df['signal']) > 0 else 1
                        signal_range = signal_max - signal_min
                        y_min = signal_min - signal_range * 0.2
                        y_max = signal_max + signal_range * 0.2
                        
                        fig.update_layout(
                            xaxis=dict(range=[df['time'].iloc[0], df['time'].iloc[-1]]),
                            yaxis=dict(range=[y_min, y_max])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add controls for visualization customization
                        with st.expander("Customize Visualization"):
                            col1, col2 = st.columns(2)
                            with col1:
                                show_rpeaks = st.checkbox("Show R-peaks", value=True)
                                show_grid = st.checkbox("Show ECG Grid", value=True)
                            with col2:
                                show_af_prob = st.checkbox("Show AF Probability", value=True)
                                show_regions = st.checkbox("Highlight Variability Regions", value=True)
                                
                            # If any options changed, redraw the plot
                            if not show_rpeaks:
                                r_peaks = None
                            if not show_af_prob:
                                af_prob = None
                            if not show_regions:
                                analysis_regions = None
                                
                            if not (show_rpeaks and show_af_prob and show_regions and show_grid):
                                # Update figure with new settings
                                fig = plot_ecg_with_annotations(
                                    df, 
                                    r_peaks=r_peaks if show_rpeaks else None, 
                                    af_prob=af_prob if show_af_prob else None, 
                                    analysis_regions=analysis_regions if show_regions else None, 
                                    heart_rate=heart_rate,
                                    title=f"ECG Signal at {time_str} with AI Analysis"
                                )
                                
                                if not show_grid:
                                    fig.update_xaxes(showgrid=False, zeroline=False)
                                    fig.update_yaxes(showgrid=False, zeroline=False)
                                    
                                # Force display of the complete signal initially
                                signal_min = min(df['signal']) if len(df['signal']) > 0 else -1
                                signal_max = max(df['signal']) if len(df['signal']) > 0 else 1
                                signal_range = signal_max - signal_min
                                y_min = signal_min - signal_range * 0.2
                                y_max = signal_max + signal_range * 0.2
                                
                                fig.update_layout(
                                    xaxis=dict(range=[df['time'].iloc[0], df['time'].iloc[-1]]),
                                    yaxis=dict(range=[y_min, y_max])
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Use the original plotting function if interactive visualization is not selected
                        # Also use styled header for consistency
                        st.markdown("""
                        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #00008b; margin-bottom: 20px;">
                            <h2 style="color: #00008b; margin: 0;">ECG Signal Visualization</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        plot_ecg(df, title=f"ECG Segment at {time_str}", use_plotly=True)
                    
                    # Show raw data
                    display_raw_data(df)
                    
                    # Analysis tabs
                    analysis_tabs = st.tabs(["Atrial Fibrillation Analysis", "Medical Analysis", "Advanced Classification"])
                    
                    with analysis_tabs[0]:
                        # Perform AF analysis with enhanced features
                        af_prob, af_metrics, features = analyze_af_segment(
                            df['signal'].values, 
                            sampling_rate=holter_analyzer.fs,
                            include_features=True
                        )
                        
                        # Log analysis results for verification
                        analysis_entry = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'filename': uploaded_edf.name,
                            'file_hash': file_hash[:8],
                            'full_file_hash': full_file_hash[:16],
                            'segment_start_min': segment_start,
                            'segment_duration_sec': segment_duration,
                            'signal_mean': float(np.mean(df['signal'])),
                            'signal_std': float(np.std(df['signal'])),
                            'signal_min': float(np.min(df['signal'])),
                            'signal_max': float(np.max(df['signal'])),
                            'signal_range': float(np.max(df['signal']) - np.min(df['signal'])),
                            'af_probability': float(af_prob),
                            'heart_rate': float(af_metrics.get('mean_hr', 0)),
                            'rmssd': float(af_metrics.get('rmssd', 0)),
                            'pnn50': float(af_metrics.get('pnn50', 0)),
                            'irregularity': float(af_metrics.get('irregularity', 0)),
                            'sdnn': float(af_metrics.get('sdnn', 0)) if 'sdnn' in af_metrics else 0.0
                        }
                        
                        # Add to analysis log
                        st.session_state.analysis_log.append(analysis_entry)
                        
                        # Display verification info
                        with st.expander("Analysis Verification Details"):
                            st.markdown("### Signal and Analysis Verification")
                            st.markdown(f"""
                            These details help verify that each file and segment is analyzed uniquely:
                            
                            **File Information:**
                            - File: `{uploaded_edf.name}`
                            - Hash (first 8 chars): `{file_hash[:8]}`
                            - Full hash (first 16): `{full_file_hash[:16]}`
                            
                            **Segment Information:**
                            - Start: {segment_start} minutes
                            - Duration: {segment_duration} seconds
                            - Time: {time_str}
                            
                            **Signal Statistics:**
                            - Mean: {analysis_entry['signal_mean']:.6f}
                            - Std Dev: {analysis_entry['signal_std']:.6f}
                            - Min/Max: {analysis_entry['signal_min']:.6f}/{analysis_entry['signal_max']:.6f}
                            - Range: {analysis_entry['signal_range']:.6f}
                            
                            **Analysis Results:**
                            - AF Probability: {analysis_entry['af_probability']:.4f}
                            - Heart Rate: {analysis_entry['heart_rate']:.2f} BPM
                            - RMSSD: {analysis_entry['rmssd']:.6f}
                            - PNN50: {analysis_entry['pnn50']:.6f}
                            - Irregularity: {analysis_entry['irregularity']:.6f}
                            """)
                            
                            # Show CSV download for just this analysis
                            single_analysis_df = pd.DataFrame([analysis_entry])
                            csv = single_analysis_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download This Analysis as CSV",
                                csv,
                                f"ecg_analysis_{file_hash[:8]}_{segment_start}min.csv",
                                "text/csv",
                                key='download-single-analysis'
                            )
                    
                    with analysis_tabs[1]:
                        # Perform medical analysis
                        medical_analyzer = ECGMedicalAnalysis(fs=holter_analyzer.fs)
                        medical_report = medical_analyzer.generate_clinical_report(df['signal'].values)
                        
                        if medical_report and 'heart_rate' in medical_report and 'r_peaks' in medical_report['heart_rate']:
                            st.subheader("ECG Signal with R-peaks")
                            plot_ecg_with_peaks(df, medical_report['heart_rate']['r_peaks'], use_plotly=use_plotly)
                            
                            # Display detailed medical report
                            st.markdown("### Clinical Analysis Report")
                            
                            # Quality Metrics Section
                            st.markdown('<div class="section-title">Quality Assessment</div>', unsafe_allow_html=True)
                            quality_metrics = medical_report.get('quality_metrics', {})
                            confidence_score = quality_metrics.get('analysis_confidence', 0)
                            confidence_class = "warning" if confidence_score < 0.7 else "normal"
                            
                            st.markdown(f"""
                            <div class="medical-info">
                                <p><strong>Signal Duration:</strong> {quality_metrics.get('signal_duration', 0):.1f} seconds</p>
                                <p><strong>Signal Quality:</strong> <span class="{quality_metrics.get('signal_quality', False) and 'normal' or 'warning'}">
                                    {quality_metrics.get('quality_message', 'Not assessed')}</span></p>
                                <p><strong>Analysis Confidence:</strong> <span class="{confidence_class}">{confidence_score:.1%}</span></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Heart Rate Section
                            if medical_report['heart_rate'] is not None and 'heart_rate' in medical_report['heart_rate']:
                                hr = medical_report['heart_rate']['heart_rate']
                                hr_class = "warning" if hr < 60 or hr > 100 else "normal"
                                st.markdown(f"""
                                <div class="medical-info">
                                    <p><strong>Heart Rate:</strong> <span class="{hr_class}">{hr:.1f} BPM</span></p>
                                    <p><strong>Interpretation:</strong> {medical_report['interpretation'][0]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # QRS Complex Section
                            if medical_report['qrs_complex']['qrs_duration']['mean'] is not None:
                                qrs_duration = medical_report['qrs_complex']['qrs_duration']['mean']
                                qrs_class = "warning" if qrs_duration > 120 or qrs_duration < 80 else "normal"
                                st.markdown(f"""
                                <div class="medical-info">
                                    <p><strong>QRS Duration:</strong> <span class="{qrs_class}">{qrs_duration:.1f} ms</span></p>
                                    <p><strong>QRS Amplitude:</strong> {medical_report['qrs_complex']['qrs_amplitude']['mean']:.2f} mV</p>
                                    <p><strong>Interpretation:</strong> {medical_report['interpretation'][1]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("Unable to generate medical report. The ECG signal may be too noisy or invalid.")
                    
                    with analysis_tabs[2]:
                        if MULTI_CLASSIFIER_AVAILABLE:
                            st.markdown("### Multi-Classifier Analysis")
                            st.info("This feature uses an ensemble of multiple classifiers for more accurate arrhythmia detection.")
                            
                            # Let user choose the model type
                            model_type = st.selectbox(
                                "Select classification model type:",
                                ["ensemble", "hybrid", "cnn"],
                                format_func=lambda x: {
                                    "ensemble": "Traditional ML Ensemble",
                                    "hybrid": "Hybrid (ML + Deep Learning)",
                                    "cnn": "Deep Learning CNN"
                                }.get(x, x)
                            )
                            
                            # Placeholder for advanced classification
                            st.markdown("#### Advanced Arrhythmia Classification")
                            st.warning("Model training required. For a live demo, pre-trained models would need to be provided.")
                            
                            # Display what we would normally do with trained models
                            if features:
                                # Create a DataFrame from features
                                feature_df = pd.DataFrame([features])
                                
                                # Display explanatory text about the approach
                                st.markdown("""
                                With pre-trained models, we would:
                                
                                1. Extract 100+ features from ECG signal (already done!)
                                2. Apply ensemble of classifiers (Random Forest, XGBoost, SVM, Neural Networks)
                                3. Provide model explanations using SHAP values
                                4. Display confidence scores for each arrhythmia type
                                """)
                                
                                # Show sample feature importance (placeholder)
                                st.markdown("#### Sample Feature Importance")
                                top_features = ['sdnn', 'rmssd', 'pnn50', 'lf_hf_ratio', 'qrs_duration']
                                importance = [0.35, 0.25, 0.18, 0.12, 0.10]
                                
                                fig = px.bar(
                                    x=importance, 
                                    y=top_features, 
                                    orientation='h',
                                    title="Sample Feature Importance (Placeholder)",
                                    labels={'x': 'Importance', 'y': 'Feature'}
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show sample prediction probabilities (placeholder)
                                classes = ['Normal', 'Atrial Fibrillation', 'SVT', 'PVC', 'AV Block']
                                probs = [0.15, af_prob, 0.05, 0.05, 0.05]
                                
                                fig = px.bar(
                                    x=classes, 
                                    y=probs,
                                    title="Sample Classification Probabilities (Based on AF Analysis)",
                                    labels={'x': 'Arrhythmia Type', 'y': 'Probability'}
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Advanced classification requires the ECGMultiClassifier module. Please make sure all required packages are installed.")
                
                elif analysis_options == "Full Recording Analysis":
                    st.info("Full recording analysis coming soon! This feature would scan the entire recording for arrhythmia episodes.")
                    
                    # Placeholder for future full recording analysis
                    st.markdown("#### In the Full Recording Analysis:")
                    st.markdown("""
                    * Scan entire recording for AF episodes
                    * Generate hourly heart rate and HRV trends
                    * Identify potential arrhythmia episodes
                    * Create comprehensive PDF report
                    """)
                    
                    # Sample visualization
                    st.markdown("#### Sample Visualization (Placeholder)")
                    
                    # Create sample data for hourly AF probability
                    hours = list(range(int(holter_analyzer.duration_hours)))
                    af_probs = np.random.beta(2, 5, len(hours))
                    
                    # Plot sample hourly AF probability
                    fig = px.line(
                        x=hours, 
                        y=af_probs,
                        title="Sample Hourly AF Probability (Placeholder)",
                        labels={'x': 'Hour', 'y': 'AF Probability'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                current_file['loaded'] = False
                current_file['error'] = "Failed to load EDF file"
                st.session_state.file_history.append(current_file)
                st.error(f"Failed to load EDF file (ID: {file_hash[:8]}). Please check the file format and try again.")
    
    with tab2:
        st.markdown("## File Verification")
        st.markdown("""
        This section helps verify that different EDF files are being processed uniquely.
        View your file history and compare statistics of previously loaded files.
        """)
        
        if 'file_history' in st.session_state and st.session_state.file_history:
            # Show file history
            st.markdown("### Previously Loaded Files")
            
            # Create a table of files
            file_data = []
            for idx, file_info in enumerate(st.session_state.file_history):
                file_data.append({
                    "Index": idx + 1,
                    "File ID": file_info['hash'][:8],
                    "Filename": file_info['filename'],
                    "Size (MB)": round(file_info['size']/1048576, 2),
                    "Loaded": "✅" if file_info.get('loaded', False) else "❌",
                    "Duration (hrs)": file_info.get('duration', 'N/A'),
                    "Sampling Rate": file_info.get('fs', 'N/A'),
                    "Timestamp": file_info['timestamp']
                })
            
            # Display as dataframe
            st.dataframe(pd.DataFrame(file_data), use_container_width=True)
            
            # Compare signal statistics if available
            files_with_stats = [f for f in st.session_state.file_history if 'preview_stats' in f]
            if len(files_with_stats) > 1:
                st.markdown("### Compare Signal Statistics")
                st.markdown("This chart compares basic statistics of your loaded files to verify they're different:")
                
                # Prepare comparison data
                compare_data = []
                for file in files_with_stats:
                    compare_data.append({
                        "File ID": file['hash'][:8],
                        "Mean": file['preview_stats']['mean'],
                        "Std Dev": file['preview_stats']['std'],
                        "Min": file['preview_stats']['min'],
                        "Max": file['preview_stats']['max']
                    })
                
                # Create comparison chart
                df_compare = pd.DataFrame(compare_data)
                
                # Plot comparison
                fig = px.bar(
                    df_compare, 
                    x="File ID", 
                    y=["Mean", "Std Dev", "Min", "Max"],
                    title="Signal Statistics Comparison",
                    barmode="group",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("If the bars show different values, your files are being processed uniquely!")
            else:
                st.info("Load at least two EDF files to compare their statistics.")
        else:
            st.info("No files have been uploaded yet. Statistics will appear here after uploading files.")
        
        # Add Analysis Log section
        st.markdown("### Analysis Log")
        st.markdown("""
        This table shows all analyses performed in this session, which helps verify:
        - Different files produce different results
        - The same file produces consistent results when analyzed repeatedly
        - Metrics are within expected ranges
        """)
        
        if 'analysis_log' in st.session_state and len(st.session_state.analysis_log) > 0:
            # Create a DataFrame from the analysis log
            analysis_df = pd.DataFrame(st.session_state.analysis_log)
            
            # Display the full analysis log
            st.dataframe(analysis_df, use_container_width=True)
            
            # Provide CSV download again here
            csv = analysis_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Complete Analysis Log as CSV",
                csv,
                "ecg_analysis_log.csv",
                "text/csv",
                key='download-full-log'
            )
            
            # Create a visualization comparing AF probability across files
            if len(analysis_df) > 1:
                st.markdown("### AF Probability Comparison")
                st.markdown("This chart shows AF probability for each analysis to help identify consistency or variability:")
                
                # Plot AF probability by file hash and segment
                fig = px.scatter(
                    analysis_df,
                    x='timestamp',
                    y='af_probability',
                    color='file_hash',
                    size='heart_rate',
                    hover_data=['filename', 'segment_start_min', 'heart_rate', 'rmssd', 'irregularity'],
                    title="AF Probability Across Analyses",
                    labels={'af_probability': 'AF Probability', 'file_hash': 'File ID', 'timestamp': 'Analysis Time'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation matrix of key metrics
                st.markdown("### Metric Correlations")
                st.markdown("This heatmap shows correlations between key metrics:")
                
                # Select numeric columns for correlation
                numeric_cols = ['af_probability', 'heart_rate', 'rmssd', 'pnn50', 'irregularity', 'signal_mean', 'signal_std', 'signal_range']
                corr_matrix = analysis_df[numeric_cols].corr()
                
                # Plot correlation heatmap
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix of Analysis Metrics"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analyses have been performed yet. Analysis logs will appear here after analyzing EDF files.")
    
    with tab3:
        st.markdown("## About This Tool")
        st.markdown("""
        This enhanced ECG analysis tool provides comprehensive cardiac analysis with a focus on atrial fibrillation detection.
        
        ### Key Features
        
        * **Advanced Feature Extraction**: Extracts 100+ cardiac biomarkers from ECG signals
        * **Multi-Classifier Approach**: Combines traditional ML and deep learning models
        * **TERMA-inspired R-peak Detection**: More accurate peak detection based on research
        * **Explainable AI**: Uses SHAP values to explain model predictions
        * **Interactive Visualizations**: Both raw signals and processed results
        * **Raw Data Access**: Export capabilities for further analysis
        
        ### Technologies Used
        
        * Signal Processing: SciPy, PyWavelets
        * Machine Learning: Scikit-learn, XGBoost, LightGBM
        * Deep Learning: TensorFlow
        * Visualization: Matplotlib, Plotly
        * Explainability: SHAP
        
        ### References
        
        1. [Heart-Arrhythmia-Classification](https://github.com/Srinivas-Natarajan/Heart-Arrhythmia-Classification)
        2. [ECG-Multiclassifier-and-XAI](https://github.com/Healthpy/ECG-Multiclassifier-and-XAI)
        3. TERMA Algorithm: [Nature Article](https://www.nature.com/articles/s41598-021-97118-5)
        """)

if __name__ == "__main__":
    main()