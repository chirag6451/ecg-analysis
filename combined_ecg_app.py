import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import tempfile
import io
import scipy.signal as sp_signal

# Import project modules
try:
    from ecg_holter_analysis import HolterAnalyzer
    from ecg_arrhythmia_classification import ECGArrhythmiaClassifier
    from ecg_medical_analysis import ECGMedicalAnalysis
    modules_available = True
except ImportError:
    st.error("Could not import ECG analysis modules. Make sure you're in the correct directory.")
    modules_available = False

# Page config
st.set_page_config(
    page_title="Combined ECG Analysis App",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS that combines styles from both apps
st.markdown("""
<style>
/* Dashboard styles */
.metric-card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
}
.metric-card h3 {
    margin-top: 0;
    color: #2c3e50;
    font-size: 1.1rem;
}
.metric-value {
    color: #2c3e50;
    font-size: 1.8rem;
    font-weight: 600;
}
.metric-unit {
    color: #7f8c8d;
    font-size: 0.9rem;
}
.metric-warning {
    color: #e74c3c;
}
.metric-normal {
    color: #27ae60;
}
.metric-caution {
    color: #f39c12;
}
.small-text {
    font-size: 0.8rem;
    color: #7f8c8d;
}

/* Streamlit app styles */
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
</style>
""", unsafe_allow_html=True)

# Helper functions from ecg_streamlit_app.py
def plot_ecg(df, title="ECG Signal"):
    """Plot ECG signal using Streamlit's native plotting."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['time'], df['signal'])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

def plot_ecg_with_peaks(df, r_peaks=None, title="ECG Signal"):
    """Plot ECG signal with R-peaks marked."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['time'], df['signal'], label='ECG Signal')
    
    if r_peaks is not None:
        ax.scatter(df['time'].iloc[r_peaks], df['signal'].iloc[r_peaks], 
                  color='red', label='R-peaks', s=50)
    
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def display_medical_report(report):
    """Display medical analysis report in a user-friendly format."""
    if report is None:
        st.error("Unable to generate medical report. The ECG signal may be too noisy or invalid.")
        return
        
    if 'error' in report:
        st.error(f"Analysis Error: {report['error']}")
        return
        
    st.markdown("### Clinical Analysis Report")
    
    # Quality Metrics Section
    st.markdown('<div class="section-title">Quality Assessment</div>', unsafe_allow_html=True)
    quality_metrics = report.get('quality_metrics', {})
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
    st.markdown('<div class="section-title">Heart Rate Analysis</div>', unsafe_allow_html=True)
    if report['heart_rate'] is not None and 'heart_rate' in report['heart_rate']:
        hr = report['heart_rate']['heart_rate']
        hr_class = "warning" if hr < 60 or hr > 100 else "normal"
        st.markdown(f"""
        <div class="medical-info">
            <p><strong>Heart Rate:</strong> <span class="{hr_class}">{hr:.1f} BPM</span></p>
            <p><strong>Interpretation:</strong> {report['interpretation'][0]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # QRS Complex Section
    if report['qrs_complex']['qrs_duration']['mean'] is not None:
        st.markdown('<div class="section-title">QRS Complex Analysis</div>', unsafe_allow_html=True)
        qrs_duration = report['qrs_complex']['qrs_duration']['mean']
        qrs_class = "warning" if qrs_duration > 120 or qrs_duration < 80 else "normal"
        st.markdown(f"""
        <div class="medical-info">
            <p><strong>QRS Duration:</strong> <span class="{qrs_class}">{qrs_duration:.1f} ms</span></p>
            <p><strong>QRS Amplitude:</strong> {report['qrs_complex']['qrs_amplitude']['mean']:.2f} mV</p>
        </div>
        """, unsafe_allow_html=True)

def analyze_af_segment(signal_data, sampling_rate=200):
    """Analyze ECG segment for atrial fibrillation."""
    if not modules_available:
        st.error("Module dependencies not available. Cannot perform AF analysis.")
        return 0, {}
    
    classifier = ECGArrhythmiaClassifier()
    
    # Run AF detection
    af_prob, af_metrics = classifier.detect_af(signal_data, sampling_rate=sampling_rate)
    
    # Display AF probability and classification
    st.markdown("### Atrial Fibrillation Detection")
    af_class = "Low Risk" if af_prob < 0.3 else "Medium Risk" if af_prob < 0.7 else "High Risk"
    af_color = "normal" if af_prob < 0.3 else "caution" if af_prob < 0.7 else "warning"
    
    st.markdown(f"""
    <div class="medical-info">
        <p><strong>AF Probability:</strong> <span class="{af_color}">{af_prob:.1%}</span></p>
        <p><strong>Classification:</strong> <span class="{af_color}">{af_class}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display metrics
    st.markdown("### Heart Rate Variability Metrics")
    metrics_md = ""
    for metric, value in af_metrics.items():
        if isinstance(value, (int, float)):
            metrics_md += f"<p><strong>{metric.upper()}:</strong> {value:.4f}</p>\n"
    
    st.markdown(f"""
    <div class="medical-info">
        {metrics_md}
    </div>
    """, unsafe_allow_html=True)
    
    return af_prob, af_metrics

# Helper functions from ecg_dashboard.py
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

def plot_ecg_timeline(df, events=None, condition_spans=None):
    """Plot ECG timeline with events and condition spans"""
    fig = go.Figure()
    
    # Check if DataFrame is valid
    if df is None or len(df) == 0:
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No ECG data to display",
            showarrow=False,
            font=dict(color="red", size=18)
        )
        fig.update_layout(
            title="Error: No ECG Data",
            height=300
        )
        return fig
    
    # Make a copy to avoid modifying original data
    df_display = df.copy()
    
    # Fix signal issues that could prevent visualization
    fixed_signal = fix_signal_issues(df['signal'].values)
    df_display['signal'] = fixed_signal
    
    # Debug info
    signal_range = np.max(fixed_signal) - np.min(fixed_signal)
    fixed_needed = not np.array_equal(fixed_signal, df['signal'].values)
    
    debug_info = f"Range: {signal_range:.6f}, Fixed: {fixed_needed}"
    print(f"ECG Timeline Debug: {debug_info}")
    
    # Add ECG signal with enhanced reliability
    fig.add_trace(go.Scatter(
        x=df_display['time'],
        y=df_display['signal'],
        mode='lines',
        name='ECG Signal',
        line=dict(
            color='rgba(0,0,0,0.8)', 
            width=1.5,  # Slightly thicker for better visibility
            shape='linear'  # Linear (not spline) for better reliability
        )
    ))
    
    # Add condition spans if provided
    if condition_spans:
        for span in condition_spans:
            # Filter to just the data in this span for min/max
            span_df = df_display[(df_display['time'] >= span['start']) & (df_display['time'] <= span['end'])]
            if not span_df.empty:
                y_min, y_max = span_df['signal'].min(), span_df['signal'].max()
                
                # Add a colored background to indicate the condition
                color_map = {
                    'AF': 'rgba(231, 76, 60, 0.2)',  # Red for AF
                    'Bradycardia': 'rgba(241, 196, 15, 0.2)',  # Yellow for bradycardia
                    'Tachycardia': 'rgba(230, 126, 34, 0.2)',  # Orange for tachycardia
                    'Normal': 'rgba(46, 204, 113, 0.2)'  # Green for normal
                }
                color = color_map.get(span['condition'], 'rgba(149, 165, 166, 0.2)')
                
                # Create the highlight span
                fig.add_trace(go.Scatter(
                    x=[span['start'], span['start'], span['end'], span['end']],
                    y=[y_min, y_max, y_max, y_min],
                    fill="toself",
                    fillcolor=color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name=span['condition'],
                    showlegend=False,
                    hoverinfo="skip"
                ))
    
    # Add events if provided
    if events:
        for event in events:
            # Add a vertical line for the event
            fig.add_shape(
                type="line",
                x0=event['time'],
                y0=df_display['signal'].min(),
                x1=event['time'],
                y1=df_display['signal'].max(),
                line=dict(color="red", width=1, dash="dot")
            )
            
            # Add a hover annotation
            fig.add_annotation(
                x=event['time'],
                y=df_display['signal'].max(),
                text=event['type'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="red"
            )
    
    # Update layout
    fig.update_layout(
        title="ECG Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 0, 0, 0.1)',
            dtick=0.2,  # Major gridlines every 0.2 seconds
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(255, 0, 0, 0.05)',
                dtick=0.04  # Minor gridlines every 0.04 seconds
            )
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 0, 0, 0.1)',
            dtick=0.5,  # Major gridlines every 0.5 mV
            minor=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(255, 0, 0, 0.05)',
                dtick=0.1  # Minor gridlines every 0.1 mV
            )
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add small debug annotation if signal was fixed
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
    
    return fig

def plot_heart_rate_trend(data):
    """Plot heart rate trend"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['hr'],
        mode='lines',
        name='Heart Rate',
        line=dict(color='rgba(231, 76, 60, 1)', width=2)
    ))
    
    # Add reference lines for different HR zones
    fig.add_shape(
        type="line",
        x0=min(data['time']),
        y0=60,
        x1=max(data['time']),
        y1=60,
        line=dict(color="rgba(241, 196, 15, 0.7)", width=1, dash="dash"),
        name="Bradycardia Threshold"
    )
    
    fig.add_shape(
        type="line",
        x0=min(data['time']),
        y0=100,
        x1=max(data['time']),
        y1=100,
        line=dict(color="rgba(230, 126, 34, 0.7)", width=1, dash="dash"),
        name="Tachycardia Threshold"
    )
    
    # Update layout
    fig.update_layout(
        title="Heart Rate Trend",
        xaxis_title="Time (s)",
        yaxis_title="Heart Rate (BPM)",
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_lorenz(rr_intervals):
    """Plot Lorenz/Poincaré plot of RR intervals"""
    if len(rr_intervals) < 2:
        return None
    
    fig = go.Figure()
    
    # Plot RR(i) vs RR(i+1)
    fig.add_trace(go.Scatter(
        x=rr_intervals[:-1],
        y=rr_intervals[1:],
        mode='markers',
        marker=dict(
            color='rgba(231, 76, 60, 0.7)',
            size=8
        ),
        name='RR Intervals'
    ))
    
    # Update layout
    fig.update_layout(
        title="Poincaré Plot",
        xaxis_title="RR(i) (s)",
        yaxis_title="RR(i+1) (s)",
        hovermode="closest",
        height=300,
        margin=dict(l=10, r=10, t=50, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    
    return fig

def calculate_af_metrics(signal, fs, r_peaks=None):
    """Calculate AF metrics from signal and detected R-peaks"""
    if modules_available:
        try:
            classifier = ECGArrhythmiaClassifier()
            af_prob, af_metrics = classifier.detect_af(signal, sampling_rate=fs)
            
            # Add RR intervals if R-peaks were provided
            if r_peaks is not None and len(r_peaks) > 1:
                af_metrics['rr_intervals'] = np.diff(r_peaks) / fs
            
            return af_prob, af_metrics
        except Exception as e:
            st.error(f"Error calculating AF metrics: {str(e)}")
    
    return 0, {}

def detect_conditions(signal, fs, r_peaks):
    """Detect conditions (AF, bradycardia, tachycardia) from signal and R-peaks"""
    conditions = []
    
    # Calculate heart rate from R-peaks (if available)
    if r_peaks is not None and len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
        
        # Detect bradycardia (HR < 60 BPM)
        if heart_rate < 60:
            conditions.append('Bradycardia')
        
        # Detect tachycardia (HR > 100 BPM)
        if heart_rate > 100:
            conditions.append('Tachycardia')
    
    # Detect AF
    af_prob, _ = calculate_af_metrics(signal, fs, r_peaks)
    if af_prob > 0.7:
        conditions.append('AF')
    
    return conditions

def main():
    st.title("Combined ECG Analysis App")
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Application Mode",
        ["Upload & Preprocess", "ECG Signal Analysis", "ECG Dashboard"]
    )
    
    # Initialize session state to store data between pages
    if 'holter_analyzer' not in st.session_state:
        st.session_state.holter_analyzer = None
    if 'segment_df' not in st.session_state:
        st.session_state.segment_df = None
    if 'fs' not in st.session_state:
        st.session_state.fs = 200
    
    # Upload & Preprocess page
    if app_mode == "Upload & Preprocess":
        st.header("Upload ECG Data")
        st.write("Upload an EDF file to begin analysis.")
        
        uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])
        
        if uploaded_file is not None:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            try:
                if modules_available:
                    # Load the EDF file
                    holter_analyzer = HolterAnalyzer()
                    
                    if holter_analyzer.load_edf_file(tmp_path):
                        st.success(f"EDF file loaded successfully: {holter_analyzer.duration_hours:.2f} hours")
                        st.session_state.holter_analyzer = holter_analyzer
                        st.session_state.fs = holter_analyzer.fs
                        
                        # Store a sample segment to display
                        segment_start = 0  # Start at beginning
                        segment_duration = 60  # 60 seconds
                        
                        # Get the segment
                        segment_df = holter_analyzer.get_segment(segment_start, segment_duration)
                        if segment_df is not None:
                            st.session_state.segment_df = segment_df
                            
                            # Show preview
                            st.subheader("ECG Signal Preview")
                            plot_ecg(segment_df, title="First 60 seconds of ECG recording")
                            
                            st.success("Data loaded successfully. Use the sidebar to navigate to analysis tools.")
                    else:
                        st.error("Failed to load EDF file. The file may not be in the correct format.")
                else:
                    st.error("ECG modules are not available. Cannot proceed with analysis.")
                
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            except Exception as e:
                st.error(f"Error processing EDF file: {str(e)}")
                
    # ECG Signal Analysis page
    elif app_mode == "ECG Signal Analysis":
        st.header("ECG Signal Analysis")
        
        if st.session_state.holter_analyzer is None:
            st.warning("Please upload an EDF file first on the 'Upload & Preprocess' page.")
            return
            
        holter_analyzer = st.session_state.holter_analyzer
        
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
            time_str = (holter_analyzer.start_time + timedelta(minutes=segment_start)).strftime("%H:%M:%S")
        else:
            time_str = f"{segment_start // 60:02d}:{segment_start % 60:02d}:00"
            
        st.markdown(f"### Analyzing segment at: **{time_str}**")
        
        # Analysis tabs
        tabs = st.tabs(["ECG Signal", "Atrial Fibrillation", "Medical Analysis"])
        
        with tabs[0]:
            plot_ecg(df, title=f"ECG Segment at {time_str}")
        
        with tabs[1]:
            # Perform AF analysis
            analyze_af_segment(df['signal'].values, sampling_rate=holter_analyzer.fs)
            
            # Show peaks
            if modules_available:
                classifier = ECGArrhythmiaClassifier()
                # Fix: Use the correct method to preprocess signal and get R-peaks
                signal_preprocessed = classifier.preprocess_signal_for_rpeaks(df['signal'].values, sampling_rate=holter_analyzer.fs)
                _, r_peaks = sp_signal.find_peaks(signal_preprocessed, distance=int(holter_analyzer.fs*0.3))
                
                if r_peaks is not None and len(r_peaks) > 0:
                    st.subheader("R-Peak Detection")
                    plot_ecg_with_peaks(df, r_peaks, title="ECG with R-peaks")
                    
                    # Calculate and display RR intervals
                    if len(r_peaks) > 1:
                        rr_intervals = np.diff(r_peaks) / holter_analyzer.fs
                        st.write(f"Average RR interval: {np.mean(rr_intervals):.4f} s")
                        st.write(f"Heart rate: {60/np.mean(rr_intervals):.1f} BPM")
                        
                        # Display Poincaré plot
                        st.subheader("Poincaré Plot (Heart Rate Variability)")
                        lorenz_fig = plot_lorenz(rr_intervals)
                        if lorenz_fig:
                            st.plotly_chart(lorenz_fig, use_container_width=True)
        
        with tabs[2]:
            # Medical analysis
            if modules_available:
                medical_analyzer = ECGMedicalAnalysis(fs=holter_analyzer.fs)
                medical_report = medical_analyzer.generate_clinical_report(df['signal'].values)
                display_medical_report(medical_report)
    
    # ECG Dashboard page
    elif app_mode == "ECG Dashboard":
        st.header("ECG Dashboard")
        
        if st.session_state.holter_analyzer is None:
            st.warning("Please upload an EDF file first on the 'Upload & Preprocess' page.")
            return
            
        holter_analyzer = st.session_state.holter_analyzer
        
        # Time selection
        st.subheader("Select Time Point")
        minutes = int(holter_analyzer.duration_hours * 60)
        selected_minute = st.slider("Time (minutes)", 0, max(1, minutes-1), 0)
        selected_time = f"{selected_minute//60:02d}:{selected_minute%60:02d}:00"
        
        # Get data segment for the selected time
        st.subheader(f"ECG Dashboard at: {selected_time}")
        segment_duration = 60  # 60 seconds
        try:
            df = holter_analyzer.get_segment(selected_minute, segment_duration)
            fs = holter_analyzer.fs
            
            # Process signal for better visualization
            signal = df['signal'].values
            
            # Create classifier if not exists
            if 'classifier' not in st.session_state:
                st.session_state.classifier = ECGArrhythmiaClassifier()
            
            classifier = st.session_state.classifier
            
            # Preprocess signal for R-peak detection
            signal_preprocessed, _, r_peaks = classifier.preprocess_signal_for_rpeaks(signal, sampling_rate=fs)
            
            # Display the first row with 3 columns
            col1, col2, col3 = st.columns(3)
            
            # Calculate HRV metrics and AF probability
            with col1:
                avg_hr, metrics = calculate_af_metrics(signal, fs, r_peaks)
                st.metric("Avg Heart Rate", f"{avg_hr:.1f} BPM")
            
            with col2:
                af_probability = classifier.detect_af(signal, fs)[0]
                af_burden = af_probability * 100
                st.metric("AF Burden", f"{af_burden:.1f}%")
                
            with col3:
                st.metric("AF Probability", f"{af_probability:.4f}")
            
            # Display ECG Timeline
            timeline_fig = plot_ecg_timeline(df)
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Create second row with 2 columns
            col1, col2 = st.columns(2)
            
            # Plot heart rate trend
            with col1:
                hr_trend_fig = plot_heart_rate_trend(calculate_hr_trend(signal, fs, r_peaks))
                st.plotly_chart(hr_trend_fig, use_container_width=True)
            
            # Plot Poincaré plot
            with col2:
                if len(r_peaks) > 3:
                    # Calculate RR intervals
                    rr_intervals = np.diff(r_peaks) / fs
                    poincare_fig = plot_lorenz(rr_intervals)
                    st.plotly_chart(poincare_fig, use_container_width=True)
                else:
                    st.warning("Not enough R-peaks detected for Poincaré plot")
            
            # Display detected conditions
            st.subheader("Cardiac Condition Summary")
            conditions = detect_conditions(signal, fs, r_peaks)
            
            if conditions:
                condition_text = ", ".join([f"{cond} ({prob:.2f})" for cond, prob in conditions.items()])
                st.info(f"Detected conditions: {condition_text}")
            else:
                st.info("No significant cardiac conditions detected in this segment")
                
        except Exception as e:
            st.error(f"Error processing data at {selected_time}: {str(e)}")
            st.exception(e)
        else:
            st.warning("Please upload and process an ECG file first")

if __name__ == "__main__":
    main() 