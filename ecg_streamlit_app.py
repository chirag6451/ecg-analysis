import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import tempfile
import io
import plotly.graph_objects as go
import neurokit2 as nk
from scipy import signal

from ecg_holter_analysis import HolterAnalyzer
from ecg_arrhythmia_classification import ECGArrhythmiaClassifier
from ecg_medical_analysis import ECGMedicalAnalysis

# Page config
st.set_page_config(
    page_title="ECG Holter Analysis",
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
    </style>
    """, unsafe_allow_html=True)

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

def detect_arrhythmias(ecg_signal, fs):
    """Detect arrhythmias in ECG signal."""
    try:
        # Process ECG signal
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
        
        # Extract R-peaks
        rpeaks = info['ECG_R_Peaks']
        
        if len(rpeaks) < 2:
            return {"has_arrhythmia": False, "types": [], "message": "Not enough R-peaks detected for analysis"}, None, None
            
        # Calculate RR intervals
        rr_intervals = np.diff(rpeaks) / fs
        
        # Calculate HRV metrics
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        
        # Calculate heart rate
        heart_rate = 60 / rr_intervals
        
        # Initialize arrhythmia dictionary
        arrhythmias = {
            'has_arrhythmia': False,
            'types': [],
            'metrics': {
                'mean_hr': np.mean(heart_rate),
                'sdnn': sdnn,
                'rmssd': rmssd
            }
        }
        
        # Check for atrial fibrillation
        if rmssd > 0.1 and sdnn > 0.05:
            arrhythmias['has_arrhythmia'] = True
            arrhythmias['types'].append({
                'name': 'Atrial Fibrillation',
                'probability': min(1.0, rmssd * 5),
                'evidence': f"RMSSD={rmssd:.3f}, SDNN={sdnn:.3f}"
            })
        
        # Check for bradycardia
        if np.mean(heart_rate) < 60:
            arrhythmias['has_arrhythmia'] = True
            arrhythmias['types'].append({
                'name': 'Bradycardia',
                'probability': 0.8,
                'evidence': f"Mean HR={np.mean(heart_rate):.1f} BPM"
            })
        
        # Check for tachycardia
        if np.mean(heart_rate) > 100:
            arrhythmias['has_arrhythmia'] = True
            arrhythmias['types'].append({
                'name': 'Tachycardia',
                'probability': 0.8,
                'evidence': f"Mean HR={np.mean(heart_rate):.1f} BPM"
            })
        
        # Check for high irregularity without AF
        if sdnn > 0.2 and not any(a['name'] == 'Atrial Fibrillation' for a in arrhythmias['types']):
            arrhythmias['has_arrhythmia'] = True
            arrhythmias['types'].append({
                'name': 'Irregular Rhythm',
                'probability': min(1.0, sdnn * 3),
                'evidence': f"SDNN={sdnn:.3f}"
            })
        
        return arrhythmias, rpeaks, heart_rate
    except Exception as e:
        return {"has_arrhythmia": False, "types": [], "message": f"Error in arrhythmia detection: {str(e)}"}, None, None

def plot_ecg_with_arrhythmia_markers(df, arrhythmias, rpeaks=None):
    """Plot ECG with arrhythmia markers and annotations."""
    fig = go.Figure()
    
    # Add ECG trace
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['signal'],
        mode='lines',
        name='ECG',
        line=dict(color='blue', width=1.2)
    ))
    
    # Add R-peaks if available
    if rpeaks is not None and len(rpeaks) > 0:
        fig.add_trace(go.Scatter(
            x=df['time'][rpeaks],
            y=df['signal'][rpeaks],
            mode='markers',
            name='R-peaks',
            marker=dict(color='red', size=8)
        ))
    
    # Add arrhythmia annotations
    if arrhythmias['has_arrhythmia']:
        # Add a colored background for the whole segment to indicate arrhythmia
        arrhythmia_types = [a['name'] for a in arrhythmias['types']]
        color = 'rgba(255, 0, 0, 0.1)'  # Default red for most arrhythmias
        
        if 'Atrial Fibrillation' in arrhythmia_types:
            color = 'rgba(255, 0, 0, 0.15)'  # Red for AF
        elif 'Bradycardia' in arrhythmia_types:
            color = 'rgba(0, 0, 255, 0.15)'  # Blue for bradycardia
        elif 'Tachycardia' in arrhythmia_types:
            color = 'rgba(255, 165, 0, 0.15)'  # Orange for tachycardia
        
        # Add a colored background
        fig.add_shape(
            type="rect",
            x0=min(df['time']),
            x1=max(df['time']),
            y0=min(df['signal']) - 0.2 * (max(df['signal']) - min(df['signal'])),
            y1=max(df['signal']) + 0.2 * (max(df['signal']) - min(df['signal'])),
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        )
        
        # Add text annotation for arrhythmia type
        fig.add_annotation(
            x=min(df['time']) + (max(df['time']) - min(df['time'])) * 0.1,
            y=max(df['signal']) + 0.1 * (max(df['signal']) - min(df['signal'])),
            text=", ".join(arrhythmia_types),
            showarrow=True,
            arrowhead=1,
            font=dict(size=14, color="red")
        )
    
    # Update layout
    fig.update_layout(
        title="ECG with Arrhythmia Detection",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        height=400,
        margin=dict(l=10, r=10, t=50, b=30)
    )
    
    return fig

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
        <p><strong>Analysis Components:</strong></p>
        <ul>
            <li>Heart Rate: {'✓' if quality_metrics.get('heart_rate_detected', False) else '✗'}</li>
            <li>QRS Complex: {'✓' if quality_metrics.get('qrs_complex_detected', False) else '✗'}</li>
            <li>ST Segment: {'✓' if quality_metrics.get('st_segment_detected', False) else '✗'}</li>
            <li>Rhythm: {'✓' if quality_metrics.get('rhythm_detected', False) else '✗'}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Only show detailed analysis if signal quality is acceptable
    if not quality_metrics.get('signal_quality', False):
        st.warning("Detailed analysis not available due to poor signal quality. Please ensure the ECG signal meets the minimum requirements.")
        return
        
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
    else:
        st.markdown("""
        <div class="medical-info">
            <p><strong>Heart Rate:</strong> <span class="warning">Unable to determine</span></p>
            <p><strong>Interpretation:</strong> Unable to analyze heart rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # QRS Complex Section
    st.markdown('<div class="section-title">QRS Complex Analysis</div>', unsafe_allow_html=True)
    if report['qrs_complex']['qrs_duration']['mean'] is not None:
        qrs_duration = report['qrs_complex']['qrs_duration']['mean']
        qrs_class = "warning" if qrs_duration > 120 or qrs_duration < 80 else "normal"
        st.markdown(f"""
        <div class="medical-info">
            <p><strong>QRS Duration:</strong> <span class="{qrs_class}">{qrs_duration:.1f} ms</span></p>
            <p><strong>QRS Amplitude:</strong> {report['qrs_complex']['qrs_amplitude']['mean']:.2f} mV</p>
            <p><strong>Interpretation:</strong> {report['interpretation'][1]}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="medical-info">
            <p><strong>QRS Complex:</strong> <span class="warning">Unable to analyze</span></p>
            <p><strong>Interpretation:</strong> Unable to analyze QRS complex</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ST Segment Section
    st.markdown('<div class="section-title">ST Segment Analysis</div>', unsafe_allow_html=True)
    if report['st_segment']['st_elevation']['mean'] is not None:
        st_elevation = report['st_segment']['st_elevation']['mean']
        st_class = "warning" if abs(st_elevation) > 0.1 else "normal"
        st.markdown(f"""
        <div class="medical-info">
            <p><strong>ST Elevation/Depression:</strong> <span class="{st_class}">{st_elevation:.2f} mV</span></p>
            <p><strong>Interpretation:</strong> {report['interpretation'][2]}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="medical-info">
            <p><strong>ST Segment:</strong> <span class="warning">Unable to analyze</span></p>
            <p><strong>Interpretation:</strong> Unable to analyze ST segment</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Rhythm Analysis Section
    st.markdown('<div class="section-title">Rhythm Analysis</div>', unsafe_allow_html=True)
    if report['rhythm']['is_regular'] is not None:
        rhythm_class = "warning" if not report['rhythm']['is_regular'] else "normal"
        st.markdown(f"""
        <div class="medical-info">
            <p><strong>Rhythm Regularity:</strong> <span class="{rhythm_class}">{'Regular' if report['rhythm']['is_regular'] else 'Irregular'}</span></p>
            <p><strong>RMSSD:</strong> {report['rhythm']['rmssd']:.3f} s</p>
            <p><strong>SDNN:</strong> {report['rhythm']['sdnn']:.3f} s</p>
            <p><strong>Interpretation:</strong> {report['interpretation'][3]}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="medical-info">
            <p><strong>Rhythm:</strong> <span class="warning">Unable to analyze</span></p>
            <p><strong>Interpretation:</strong> Unable to determine rhythm regularity</p>
        </div>
        """, unsafe_allow_html=True)

def analyze_af_segment(signal_data, sampling_rate=200):
    """Analyze a segment for AF detection."""
    classifier = ECGArrhythmiaClassifier()
    af_prob, af_metrics = classifier.detect_af(signal_data, sampling_rate=sampling_rate)
    
    # Display AF probability with gauge chart
    st.markdown("### Atrial Fibrillation Analysis")
    
    # Create gauge chart for AF probability
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Define color based on probability
    if af_prob < 0.3:
        color = "green"
        risk_level = "Low Risk"
    elif af_prob < 0.7:
        color = "orange"
        risk_level = "Moderate Risk"
    else:
        color = "red"
        risk_level = "High Risk"
    
    # Draw gauge chart
    ax.barh(0, af_prob, color=color, height=0.5)
    ax.barh(0, 1.0, color='lightgray', height=0.5, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title(f'AF Probability: {af_prob*100:.1f}%', fontsize=16)
    ax.text(0.5, -0.25, risk_level, ha='center', fontsize=14, fontweight='bold', color=color)
    
    st.pyplot(fig)
    
    # Display metrics in columns
    if "error" in af_metrics:
        st.error(f"Analysis Error: {af_metrics['error']}")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Heart Rate Variability Metrics")
            st.markdown(f"""
            - **RR Interval Std Dev (SDNN):** {af_metrics.get('rr_std', 0):.3f} s
            - **RMSSD:** {af_metrics.get('rmssd', 0):.3f} s
            - **PNN50:** {af_metrics.get('pnn50', 0)*100:.1f}%
            """)
            
        with col2:
            st.markdown("#### Rhythm Metrics")
            st.markdown(f"""
            - **Mean Heart Rate:** {af_metrics.get('mean_hr', 0):.1f} BPM
            - **RR Irregularity:** {af_metrics.get('irregularity', 0):.2f}
            """)
    
    # Display AF assessment
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

def analyze_af_segment_minimal(signal_data, sampling_rate=200):
    """
    Minimal version of the AF analysis function that avoids visualization issues
    
    Args:
        signal_data: ECG signal data
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Basic signal preparation
        signal = np.array(signal_data).astype(float)
        
        # Handle NaN and Inf values
        signal = np.nan_to_num(signal, nan=np.nanmean(signal) if np.any(~np.isnan(signal)) else 0)
        
        # Detect R-peaks with minimal parameters
        r_peaks, = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
        rpeaks = r_peaks['ECG_R_Peaks']
        
        # Calculate heart rate
        if len(rpeaks) >= 2:
            # Calculate RR intervals in seconds
            rr_intervals = np.diff(rpeaks) / sampling_rate
            
            # Calculate heart rate in BPM
            heart_rate = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
            
            # Basic HRV metrics
            sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else 0
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else 0
            
            # Calculate pNN50
            nn50 = sum(abs(np.diff(rr_intervals)) > 0.05)
            pnn50 = nn50 / len(rr_intervals) if len(rr_intervals) > 0 else 0
            
            # Simple regularity metric (coefficient of variation)
            irregularity = sdnn / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            
            # Very basic AF probability based on irregularity and heart rate
            base_probability = 0.5 * irregularity + 0.3 * (1 if heart_rate > 100 else 0)
            base_probability = max(0, min(1, base_probability))  # Clip to [0,1]
            
            # Apply deterministic offset based on heart rate
            deterministic_offset = 0.001 * (heart_rate - 70)
            final_probability = max(0, min(1, base_probability + deterministic_offset))
            
            # Also detect other arrhythmias
            arrhythmias, _, _ = detect_arrhythmias(signal, sampling_rate)
            
            # Prepare results
            results = {
                'heart_rate': {
                    'mean': heart_rate,
                    'r_peaks': rpeaks,
                    'rr_intervals': rr_intervals
                },
                'hrv': {
                    'sdnn': sdnn,
                    'rmssd': rmssd,
                    'pnn50': pnn50
                },
                'af_metrics': {
                    'irregularity': irregularity,
                    'base_probability': base_probability,
                    'deterministic_offset': deterministic_offset,
                    'final_probability': final_probability
                },
                'arrhythmias': arrhythmias
            }
            
            print(f"Number of R-peaks detected: {len(rpeaks)}")
            print("AF Detection Metrics:")
            print(f"  Mean HR: {heart_rate:.2f}")
            print(f"  SDNN: {sdnn:.4f}")
            print(f"  RMSSD: {rmssd:.4f}")
            print(f"  pNN50: {pnn50:.4f}")
            print(f"  Irregularity: {irregularity:.4f}")
            print("Probability Calculation:")
            print(f"  Base probability: {base_probability:.4f}")
            print(f"  Deterministic offset: {deterministic_offset:.6f}")
            print(f"  Final probability: {final_probability:.4f}")
            
            return results
        else:
            print("Not enough R-peaks detected for analysis")
            return None
    except Exception as e:
        print(f"Error in AF analysis: {str(e)}")
        return None

def main():
    st.title("ECG Holter Analysis with Atrial Fibrillation Detection")
    st.markdown("""
    This specialized tool analyzes Holter ECG recordings for atrial fibrillation and other cardiac abnormalities.
    Upload your EDF file to get comprehensive analysis and AF detection.
    """)
    
    st.info("""
    ### Large EDF File Support
    
    This app supports large Holter EDF files (30-50+ MB). For optimal performance:
    - Make sure you have installed the MNE package: `pip install mne`
    - Some EDF files may take several minutes to load, especially if they're large
    - If you encounter errors, try processing the file with EDFbrowser or other EDF tools first
    """)
    
    uploaded_edf = st.file_uploader(
        "Upload Holter EDF file",
        type=['edf']
    )
    
    if uploaded_edf:
        # Show loading status
        status_placeholder = st.empty()
        status_placeholder.info(f"Loading EDF file ({uploaded_edf.size/1048576:.1f} MB). Please wait...")
        
        # Create a temporary file to save the uploaded EDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
            tmp_file.write(uploaded_edf.getvalue())
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
                
                # Plot the segment
                plot_ecg(df, title=f"ECG Segment at {time_str}")
                
                # Analysis tabs
                tabs = st.tabs(["Atrial Fibrillation Analysis", "Medical Analysis"])
                
                with tabs[0]:
                    # Perform AF analysis
                    analyze_af_segment(df['signal'].values, sampling_rate=holter_analyzer.fs)
                
                with tabs[1]:
                    # Perform medical analysis
                    medical_analyzer = ECGMedicalAnalysis(fs=holter_analyzer.fs)
                    medical_report = medical_analyzer.generate_clinical_report(df['signal'].values)
                    
                    if medical_report and 'heart_rate' in medical_report and 'r_peaks' in medical_report['heart_rate']:
                        st.subheader("ECG Signal with R-peaks")
                        plot_ecg_with_peaks(df, medical_report['heart_rate']['r_peaks'])
                    
                    display_medical_report(medical_report)
            
            elif analysis_options == "Full Recording Analysis":
                # Analysis parameters
                segment_minutes = st.slider(
                    "Segment size (minutes)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1
                )
                
                overlap_minutes = st.slider(
                    "Overlap between segments (minutes)",
                    min_value=0,
                    max_value=segment_minutes-1,
                    value=1,
                    step=1
                )
                
                if st.button("Run Full Analysis", type="primary"):
                    # Warning about long processing times
                    st.warning(f"Analyzing {holter_analyzer.duration_hours:.1f} hours of data. This may take several minutes or even hours depending on the recording length.")
                    
                    with st.spinner("Analyzing full Holter recording... This may take several minutes."):
                        try:
                            # Run full recording analysis
                            results = holter_analyzer.analyze_full_recording(
                                segment_minutes=segment_minutes,
                                overlap_minutes=overlap_minutes
                            )
                            
                            if results:
                                # Generate report
                                report_html = holter_analyzer.generate_holter_report(results)
                                
                                # Display summary
                                st.markdown("## Holter Analysis Results")
                                
                                # Display file info
                                st.markdown("### Recording Information")
                                file_info = results['file_info']
                                st.markdown(f"""
                                - **Patient ID**: {file_info.get('patient_id', 'Unknown')}
                                - **Recording Date**: {file_info.get('recording_date', 'Unknown')}
                                - **Duration**: {file_info.get('duration', 0) / 3600:.2f} hours
                                """)
                                
                                # AF burden section
                                st.markdown("### Atrial Fibrillation Summary")
                                af_summary = results['summary'].get('atrial_fibrillation', {})
                                
                                # Create AF burden gauge
                                af_burden = af_summary.get('burden_percent', 0)
                                af_class = "normal" if af_burden < 1 else "warning"
                                
                                # Create gauge chart
                                fig, ax = plt.subplots(figsize=(8, 3))
                                ax.barh(0, min(af_burden, 100) / 100, color='red' if af_burden > 1 else 'green', height=0.5)
                                ax.barh(0, 1.0, color='lightgray', height=0.5, alpha=0.3)
                                ax.set_xlim(0, 1)
                                ax.set_ylim(-0.5, 0.5)
                                ax.set_yticks([])
                                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                                ax.set_title(f'AF Burden: {af_burden:.1f}%', fontsize=16)
                                st.pyplot(fig)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("AF Episodes Detected", af_summary.get('episodes', 0))
                                with col2:
                                    st.metric("Total AF Duration (minutes)", af_summary.get('total_minutes', 0))
                                
                                # Clinical assessment
                                if af_burden >= 5:
                                    st.error("**High AF Burden**: This recording shows significant atrial fibrillation burden which may require medical intervention.")
                                elif af_burden >= 1:
                                    st.warning("**Moderate AF Burden**: This recording shows some atrial fibrillation activity which should be evaluated by a healthcare professional.")
                                else:
                                    st.success("**Low AF Burden**: This recording shows minimal or no atrial fibrillation activity.")
                                
                                # Display summary statistics
                                st.markdown("### Cardiac Metrics")
                                summary = results['summary']
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if 'heart_rate' in summary:
                                        hr = summary['heart_rate']
                                        st.markdown(f"**Heart Rate**: {hr['mean']:.1f} BPM (Range: {hr['min']:.1f}-{hr['max']:.1f})")
                                    
                                    if 'qrs_duration' in summary:
                                        qrs = summary['qrs_duration']
                                        st.markdown(f"**QRS Duration**: {qrs['mean']:.1f} ms (Range: {qrs['min']:.1f}-{qrs['max']:.1f})")
                                
                                with col2:
                                    if 'st_elevation' in summary:
                                        st_el = summary['st_elevation']
                                        st.markdown(f"**ST Elevation**: {st_el['mean']:.2f} mV (Range: {st_el['min']:.2f}-{st_el['max']:.2f})")
                                    
                                    if 'arrhythmias' in summary:
                                        arr = summary['arrhythmias']
                                        st.markdown(f"**Arrhythmia Episodes**: {arr['episodes']}")
                                
                                # Display episodes
                                if results['af_episodes']:
                                    st.markdown("### Atrial Fibrillation Episodes")
                                    af_df = pd.DataFrame(results['af_episodes'])
                                    af_df = af_df[['time', 'duration_minutes', 'probability']]
                                    st.dataframe(af_df)
                                
                                if results['arrhythmia_episodes']:
                                    st.markdown("### Arrhythmia Episodes")
                                    # Convert to simplified dataframe
                                    arr_list = []
                                    for episode in results['arrhythmia_episodes']:
                                        arr_types = ", ".join([f"{k}: {v}" for k, v in episode['arrhythmias'].items()])
                                        arr_list.append({
                                            'time': episode['time'],
                                            'duration_minutes': episode['duration_minutes'],
                                            'arrhythmias': arr_types
                                        })
                                    arr_df = pd.DataFrame(arr_list)
                                    st.dataframe(arr_df)
                                
                                # Generate hourly heart rate plot
                                st.markdown("### Heart Rate Trend")
                                hr_data = results['hourly_heart_rates']
                                if any(len(h) > 0 for h in hr_data):
                                    fig = holter_analyzer._plot_hourly_stats(hr_data, 'Heart Rate (BPM)')
                                    if fig:
                                        st.pyplot(fig)
                                
                                # Download report
                                st.download_button(
                                    label="Download Full HTML Report",
                                    data=report_html,
                                    file_name="holter_report.html",
                                    mime="text/html"
                                )
                                
                                # Get a sample segment to display
                                if len(results['af_episodes']) > 0:
                                    # Display a segment with AF
                                    af_minute = results['af_episodes'][0]['start_minute']
                                    df = holter_analyzer.get_segment(af_minute, 60)
                                    st.markdown(f"### Sample ECG Segment (AF Episode at {results['af_episodes'][0]['time']})")
                                else:
                                    # Display first segment
                                    df = holter_analyzer.get_segment(0, 60)
                                    st.markdown("### Sample ECG Segment (First Segment)")
                                
                                # Plot the segment
                                fig, ax = plt.subplots(figsize=(12, 4))
                                ax.plot(df['time'], df['signal'])
                                ax.set_xlabel("Time (seconds)")
                                ax.set_ylabel("Amplitude")
                                ax.set_title("ECG Signal")
                                ax.grid(True)
                                st.pyplot(fig)
                                
                                # Analyze this specific segment for AF
                                st.markdown("### Detailed Analysis of Sample Segment")
                                analyze_af_segment(df['signal'].values, sampling_rate=holter_analyzer.fs)
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.info("""
                            Possible solutions:
                            1. Try analyzing a smaller portion of the recording
                            2. Increase your system's memory
                            3. Process the EDF file with EDF tools before importing
                            """)
        else:
            # Error when loading file
            status_placeholder.error("Failed to load EDF file. See details below.")
            st.error("""
            ### EDF File Loading Error
            
            The file could not be loaded. This could be due to:
            
            1. **Format Issues**: The file may not follow the EDF/EDF+ standard exactly
            2. **File Size**: Very large files may require more memory
            3. **File Structure**: Missing or corrupt header information
            
            ### Solutions:
            
            1. Make sure MNE is installed: `pip install mne`
            2. Try preprocessing your EDF file with EDF Browser or other tools
            3. Try a smaller or different EDF file
            4. Check if your file is password-protected or encrypted
            """)
        
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except Exception as e:
            print(f"Error removing temp file: {str(e)}")

if __name__ == "__main__":
    main() 