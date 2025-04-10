import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from datetime import datetime
import scipy.signal as sp_signal

# Import project modules - use try/except to handle import issues gracefully
try:
    from ecg_medical_analysis import ECGMedicalAnalysis
    from ecg_arrhythmia_classification import ECGArrhythmiaClassifier
    from ecg_holter_analysis import HolterAnalyzer
    medical_analysis_available = True
except ImportError:
    st.warning("Could not import ECG analysis modules. Some functionality may be limited.")
    medical_analysis_available = False

def plot_signal_debug(signal, fs=200, title="Signal Debugging"):
    """
    Create a detailed diagnostic plot of a signal with various metrics
    
    Args:
        signal: numpy array containing the signal
        fs: sampling frequency (Hz)
        title: plot title
    """
    # Configure subplot specs to support table in position (3,2)
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],  # Last cell is table type
        ],
        subplot_titles=("Raw Signal", "Histogram", "Power Spectrum", "Autocorrelation", 
                        "First Derivative", "Signal Statistics")
    )
    
    # Time vector
    time = np.arange(len(signal)) / fs
    
    # 1. Raw signal plot
    fig.add_trace(
        go.Scatter(x=time, y=signal, mode='lines', name='Signal'),
        row=1, col=1
    )
    
    # 2. Histogram
    fig.add_trace(
        go.Histogram(x=signal, nbinsx=50, name='Histogram'),
        row=1, col=2
    )
    
    # 3. Power spectrum
    if len(signal) > 0:
        f, psd = sp_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
        fig.add_trace(
            go.Scatter(x=f, y=10 * np.log10(psd), mode='lines', name='PSD'),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Power (dB)", row=2, col=1)
    
    # 4. Autocorrelation
    if len(signal) > 1:
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_time = np.arange(len(autocorr)) / fs
        fig.add_trace(
            go.Scatter(x=autocorr_time, y=autocorr/max(autocorr), mode='lines', name='Autocorrelation'),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Lag (s)", row=2, col=2)
    
    # 5. First derivative
    if len(signal) > 1:
        derivative = np.diff(signal)
        derivative_time = time[:-1]
        fig.add_trace(
            go.Scatter(x=derivative_time, y=derivative, mode='lines', name='Derivative'),
            row=3, col=1
        )
    
    # 6. Signal statistics as a table
    if len(signal) > 0:
        stats = {
            "Metric": ["Min", "Max", "Mean", "Median", "Std Dev", "Range", "Zero Crossings", "Signal Length"],
            "Value": [
                f"{np.min(signal):.6f}",
                f"{np.max(signal):.6f}",
                f"{np.mean(signal):.6f}",
                f"{np.median(signal):.6f}",
                f"{np.std(signal):.6f}",
                f"{np.max(signal) - np.min(signal):.6f}",
                f"{np.sum(np.diff(np.signbit(signal)) != 0)}",
                f"{len(signal)} samples ({len(signal)/fs:.2f} s)"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(stats.keys()), align='center', font=dict(size=12)),
                cells=dict(values=list(stats.values()), align='center', font=dict(size=12))
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=900, 
        width=1000,
        title_text=title,
        showlegend=False
    )
    
    return fig

def visualize_processing_steps(signal, fs=200):
    """Visualize each step of signal processing to diagnose issues"""
    if len(signal) == 0:
        st.error("Empty signal provided for processing visualization.")
        return
    
    # Create medical analyzer instance if available
    if medical_analysis_available:
        analyzer = ECGMedicalAnalysis(fs=fs)
    else:
        st.warning("Medical analysis module not available. Using simplified processing.")
        analyzer = None
    
    # Create the steps for visualization
    steps = []
    
    # Step 1: Raw signal
    steps.append({
        "name": "Raw Signal",
        "data": signal.copy(),
        "description": "Original unprocessed signal"
    })
    
    # Step 2: Normalize to zero mean and unit variance
    normalized = (signal - np.mean(signal)) / (np.std(signal) if np.std(signal) > 0 else 1.0)
    steps.append({
        "name": "Normalized Signal",
        "data": normalized,
        "description": "Signal normalized to zero mean and unit variance"
    })
    
    # Step 3: High-pass filter (remove baseline wander)
    nyquist = fs / 2
    high_cutoff = 0.5 / nyquist
    b, a = sp_signal.butter(4, high_cutoff, btype='high')
    high_passed = sp_signal.filtfilt(b, a, normalized)
    steps.append({
        "name": "High-Pass Filtered",
        "data": high_passed,
        "description": "Applied 0.5 Hz high-pass filter to remove baseline wander"
    })
    
    # Step 4: Low-pass filter (remove high-frequency noise)
    low_cutoff = 40 / nyquist
    b, a = sp_signal.butter(4, low_cutoff, btype='low')
    band_passed = sp_signal.filtfilt(b, a, high_passed)
    steps.append({
        "name": "Band-Pass Filtered",
        "data": band_passed,
        "description": "Applied 0.5-40 Hz band-pass filter"
    })
    
    # Step 5: Notch filter (remove power line interference)
    notch_freq = 50  # or 60 depending on region
    q = 30.0
    b, a = sp_signal.iirnotch(notch_freq/nyquist, q)
    notched = sp_signal.filtfilt(b, a, band_passed)
    steps.append({
        "name": "Notch Filtered",
        "data": notched,
        "description": f"Removed {notch_freq} Hz power line interference"
    })
    
    # Step 6: Moving average filter (additional smoothing)
    window_size = int(0.02 * fs)  # 20ms window
    if window_size > 1:
        moving_avg = np.convolve(notched, np.ones(window_size)/window_size, mode='same')
        steps.append({
            "name": "Moving Average Filtered",
            "data": moving_avg,
            "description": f"Applied {window_size}-point moving average filter"
        })
    else:
        moving_avg = notched
    
    # Step 7: Median filter (remove spikes)
    window_size = int(0.01 * fs)  # 10ms window
    if window_size > 1:
        window_size = window_size if window_size % 2 == 1 else window_size + 1  # Ensure odd window size
        median_filtered = sp_signal.medfilt(moving_avg, kernel_size=window_size)
        steps.append({
            "name": "Median Filtered",
            "data": median_filtered,
            "description": f"Applied {window_size}-point median filter to remove spikes"
        })
    else:
        median_filtered = moving_avg
    
    # If medical analyzer is available, also show complete processing
    if analyzer is not None:
        try:
            fully_processed = analyzer.preprocess_signal(signal)
            steps.append({
                "name": "Fully Processed",
                "data": fully_processed,
                "description": "Signal after complete ECGMedicalAnalysis preprocessing"
            })
        except Exception as e:
            st.error(f"Error in full processing pipeline: {str(e)}")
    
    # Create tabs for each processing step
    tabs = st.tabs([step["name"] for step in steps])
    
    # Fill each tab with the step visualization
    for i, (tab, step) in enumerate(zip(tabs, steps)):
        with tab:
            st.markdown(f"### {step['name']}")
            st.markdown(step["description"])
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min", f"{np.min(step['data']):.4f}")
            col2.metric("Max", f"{np.max(step['data']):.4f}")
            col3.metric("Mean", f"{np.mean(step['data']):.4f}")
            col4.metric("Std Dev", f"{np.std(step['data']):.4f}")
            
            # Show plot with Plotly
            fig = go.Figure()
            
            time = np.arange(len(step['data']))/fs
            fig.add_trace(go.Scatter(
                x=time, 
                y=step['data'],
                mode='lines', 
                name=step['name']
            ))
            
            # Compare with original if not the first step
            if i > 0:
                # Resize original to match if needed
                orig = steps[0]['data']
                if len(orig) != len(step['data']):
                    orig = orig[:len(step['data'])] if len(orig) > len(step['data']) else np.pad(orig, (0, len(step['data'])-len(orig)))
                
                fig.add_trace(go.Scatter(
                    x=time,
                    y=orig,
                    mode='lines',
                    opacity=0.3,
                    line=dict(dash='dot'),
                    name='Original'
                ))
            
            fig.update_layout(
                title=f"{step['name']} Visualization",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=400,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show additional diagnostic plots
            with st.expander("Show Detailed Diagnostics"):
                diag_fig = plot_signal_debug(step['data'], fs=fs, title=f"Diagnostics for {step['name']}")
                st.plotly_chart(diag_fig, use_container_width=True)

def test_r_peak_detection(signal, fs=200):
    """Test different R-peak detection methods and visualize results"""
    if len(signal) == 0:
        st.error("Empty signal provided for R-peak detection.")
        return
    
    # Normalize signal
    signal = (signal - np.mean(signal)) / (np.std(signal) if np.std(signal) > 0 else 1.0)
    
    # Apply bandpass filter to isolate QRS complex frequencies
    nyquist = fs / 2
    low = 5 / nyquist
    high = 20 / nyquist
    b, a = sp_signal.butter(4, [low, high], btype='band')
    filtered = sp_signal.filtfilt(b, a, signal)
    
    # Initialize detection methods
    detection_methods = {}
    
    # Method 1: Simple threshold
    threshold = 0.6 * np.max(filtered)
    min_distance = int(0.2 * fs)  # Minimum 200ms between peaks
    peaks, _ = sp_signal.find_peaks(filtered, height=threshold, distance=min_distance)
    detection_methods["Simple Threshold"] = {
        "peaks": peaks,
        "description": "Basic peak finding with amplitude threshold and minimum distance"
    }
    
    # Method 2: Derivative-based
    derivative = np.diff(filtered)
    derivative = np.append(derivative, derivative[-1])
    squared = derivative ** 2
    
    # Moving window integration
    window_size = int(0.08 * fs)  # 80ms window
    if window_size > 1:
        moving_avg = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # Dynamic threshold
        threshold = 0.3 * np.max(moving_avg)
        peaks, _ = sp_signal.find_peaks(moving_avg, height=threshold, distance=min_distance)
        detection_methods["Derivative-Based"] = {
            "peaks": peaks,
            "description": "Uses signal derivative and moving average for detection"
        }
    
    # Method 3: Using ECGMedicalAnalysis if available
    if medical_analysis_available:
        try:
            analyzer = ECGMedicalAnalysis(fs=fs)
            r_peaks = analyzer._find_r_peaks(signal)
            detection_methods["ECGMedicalAnalysis"] = {
                "peaks": r_peaks,
                "description": "Using internal _find_r_peaks method from ECGMedicalAnalysis"
            }
        except Exception as e:
            st.error(f"Error using ECGMedicalAnalysis for R-peak detection: {str(e)}")
    
    # Method 4: Using ECGArrhythmiaClassifier if available
    if medical_analysis_available:
        try:
            classifier = ECGArrhythmiaClassifier()
            preprocessed = classifier.preprocess_signal_for_rpeaks(signal, fs)
            _, r_peaks = sp_signal.find_peaks(preprocessed, distance=min_distance)
            detection_methods["ArrhythmiaClassifier"] = {
                "peaks": r_peaks,
                "description": "Using preprocess_signal_for_rpeaks from ECGArrhythmiaClassifier"
            }
        except Exception as e:
            st.error(f"Error using ECGArrhythmiaClassifier for R-peak detection: {str(e)}")
    
    # Display the results
    tabs = st.tabs(list(detection_methods.keys()))
    
    time = np.arange(len(signal))/fs
    
    for tab, (method_name, method_data) in zip(tabs, detection_methods.items()):
        with tab:
            st.markdown(f"### {method_name}")
            st.markdown(method_data["description"])
            
            peaks = method_data["peaks"]
            
            # Calculate heart rate if possible
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / fs
                heart_rate = 60 / np.mean(rr_intervals)
                st.metric("Estimated Heart Rate", f"{heart_rate:.1f} BPM")
                st.metric("Number of R-peaks", len(peaks))
                
                # Calculate heart rate variability metrics
                sdnn = np.std(rr_intervals)
                rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
                
                col1, col2 = st.columns(2)
                col1.metric("SDNN", f"{sdnn:.4f} s")
                col2.metric("RMSSD", f"{rmssd:.4f} s")
            else:
                st.warning(f"Only {len(peaks)} peaks detected - not enough to calculate heart rate")
            
            # Show plot
            fig = go.Figure()
            
            # Add ECG signal
            fig.add_trace(go.Scatter(
                x=time, 
                y=signal,
                mode='lines', 
                name='ECG Signal'
            ))
            
            # Add R-peaks
            if len(peaks) > 0:
                peak_times = [time[p] for p in peaks if p < len(time)]
                peak_amplitudes = [signal[p] for p in peaks if p < len(signal)]
                
                fig.add_trace(go.Scatter(
                    x=peak_times,
                    y=peak_amplitudes,
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='R-peaks'
                ))
            
            fig.update_layout(
                title=f"R-peak Detection using {method_name}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def generate_test_signal(fs=200, duration=10):
    """Generate a synthetic ECG signal for testing"""
    # Time vector
    t = np.arange(0, duration, 1/fs)
    
    # Base frequency of heartbeat (1 Hz = 60 BPM)
    heart_rate = 75  # BPM
    f_heart = heart_rate / 60
    
    # Parameters for a simple ECG waveform
    # Create a repeating pattern with sharper peaks for R waves
    signal = np.zeros_like(t)
    
    # Generate PQRST complexes
    for i in range(int(duration * f_heart)):
        # Position of the current heartbeat
        t_center = i / f_heart
        
        # R peak (main spike)
        r_width = 0.025
        r_amp = 1.0
        r_offset = -0.02
        signal += r_amp * np.exp(-((t - t_center - r_offset) / r_width) ** 2)
        
        # Q wave (small negative deflection before R)
        q_width = 0.03
        q_amp = -0.2
        q_offset = -0.05
        signal += q_amp * np.exp(-((t - t_center - q_offset) / q_width) ** 2)
        
        # S wave (small negative deflection after R)
        s_width = 0.03
        s_amp = -0.4
        s_offset = 0.03
        signal += s_amp * np.exp(-((t - t_center - s_offset) / s_width) ** 2)
        
        # P wave (small positive deflection before QRS)
        p_width = 0.04
        p_amp = 0.2
        p_offset = -0.12
        signal += p_amp * np.exp(-((t - t_center - p_offset) / p_width) ** 2)
        
        # T wave (small positive deflection after QRS)
        t_width = 0.06
        t_amp = 0.3
        t_offset = 0.15
        signal += t_amp * np.exp(-((t - t_center - t_offset) / t_width) ** 2)
    
    return signal

def main():
    st.set_page_config(page_title="ECG Diagnostics", page_icon="❤️", layout="wide")
    
    # Navigation - rearranged to separate doctor view from debug view
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Doctor View", "Developer/Debug Mode"]
    )
    
    if view_mode == "Doctor View":
        st.title("ECG Analysis for Clinical Use")
        st.markdown("""
        This tool provides ECG visualization and analysis optimized for clinical assessment.
        Upload an ECG recording or select a sample to begin analysis.
        """)
        
        # Data source selection for doctor view - simplified
        data_source = st.sidebar.radio(
            "Data Source",
            ["Upload ECG Recording", "Synthetic Demo Signal", "Sample from EDF File"]
        )
    else:  # Developer/Debug Mode
        st.title("ECG Signal Diagnostics - Developer Mode")
        st.markdown("""
        This tool helps diagnose issues with ECG signal visualization and processing. 
        Upload your ECG data or use a synthetic test signal to troubleshoot problems.
        """)
        
        # Navigation for debug mode
        debug_section = st.sidebar.radio(
            "Diagnostic Tools", 
            ["Signal Viewer", "Processing Pipeline", "R-peak Detection", "AF Detection Test", "Info"]
        )
        
        # Data source selection
        data_source = st.sidebar.radio(
            "Data Source",
            ["Upload File", "Synthetic Test Signal", "Sample from EDF (if available)"]
        )
    
    # Get the data
    df = None
    signal = None
    fs = 200  # Default sampling rate
    
    if "Upload" in data_source:
        file_types = ["csv", "txt"] if view_mode == "Developer/Debug Mode" else ["csv", "txt", "edf"]
        uploaded_file = st.sidebar.file_uploader(f"Upload {data_source.split()[-1]}", type=file_types)
        
        if uploaded_file is not None:
            try:
                # EDF file handling
                if uploaded_file.name.lower().endswith('.edf'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    if medical_analysis_available:
                        holter = HolterAnalyzer()
                        if holter.load_edf_file(tmp_path):
                            st.sidebar.success(f"EDF file loaded: {holter.duration_hours:.2f} hours")
                            
                            # In doctor view, get a segment
                            sample_minute = st.sidebar.slider("Start time (minute)", 0, int(holter.duration_hours * 60) - 1, 0)
                            segment_duration = 60  # Default to 60 seconds
                            
                            segment_df = holter.get_segment(sample_minute, segment_duration)
                            if segment_df is not None and len(segment_df) > 0:
                                df = segment_df
                                signal = df['signal'].values
                                fs = holter.fs
                                st.sidebar.success(f"Loaded segment at {sample_minute} minute")
                            else:
                                st.sidebar.error("Could not extract segment from EDF file")
                        else:
                            st.sidebar.error("Failed to load EDF file")
                        
                        # Clean up the temporary file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                    else:
                        st.error("Medical analysis modules not available. Cannot load EDF files.")
                # CSV/TXT file handling
                else:
                    try:
                        # Try to load as CSV first
                        df = pd.read_csv(uploaded_file)
                        
                        # Check if we have expected columns
                        if 'time' in df.columns and 'signal' in df.columns:
                            st.sidebar.success("File loaded successfully with time and signal columns")
                            signal = df['signal'].values
                        else:
                            # If not, assume the first column is time and the second is signal
                            if len(df.columns) >= 2:
                                df.columns = ['time', 'signal'] + list(df.columns[2:])
                                st.sidebar.warning("Assuming first column is time and second is signal")
                                signal = df['signal'].values
                            elif len(df.columns) == 1:
                                # Only one column - assume it's just the signal
                                signal = df.iloc[:, 0].values
                                df['time'] = np.arange(len(signal)) / fs
                                st.sidebar.warning("Only one column found - assuming it's the signal")
                    except Exception:
                        # If CSV fails, try to load as simple text file with one value per line
                        uploaded_file.seek(0)
                        lines = uploaded_file.read().decode('utf-8').strip().split('\n')
                        try:
                            signal = np.array([float(line.strip()) for line in lines])
                            df = pd.DataFrame({
                                'time': np.arange(len(signal)) / fs,
                                'signal': signal
                            })
                            st.sidebar.warning("Loaded as plain text file with one value per line")
                        except:
                            st.error("Could not parse the uploaded file. Please ensure it's a valid file.")
                            signal = None
                            
                    # Try to detect the sampling frequency
                    if df is not None and 'time' in df.columns and len(df) > 1:
                        # Calculate sampling rate from time column
                        time_diffs = np.diff(df['time'])
                        avg_diff = np.mean(time_diffs)
                        if avg_diff > 0:
                            detected_fs = 1 / avg_diff
                            fs = int(round(detected_fs))
                            st.sidebar.info(f"Detected sampling rate: {fs} Hz")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                signal = None
    
    elif "Synthetic" in data_source:
        # Options for synthetic signal
        synthetic_duration = st.sidebar.slider("Duration (seconds)", 1, 30, 10)
        synthetic_fs = st.sidebar.slider("Sampling Frequency (Hz)", 100, 1000, 200)
        
        # Generate synthetic signal
        signal = generate_test_signal(fs=synthetic_fs, duration=synthetic_duration)
        fs = synthetic_fs
        
        # Create dataframe
        df = pd.DataFrame({
            'time': np.arange(len(signal)) / fs,
            'signal': signal
        })
        
        st.sidebar.success(f"Generated synthetic ECG signal ({len(signal)} samples, {fs} Hz)")
    
    # Set sampling rate
    fs_input = st.sidebar.number_input("Sampling Rate (Hz)", min_value=1.0, max_value=2000.0, value=float(fs), step=1.0)
    fs = fs_input  # Update with user input
    
    # Process different views
    if signal is not None and len(signal) > 0:
        st.sidebar.markdown(f"**Signal length:** {len(signal)} samples ({len(signal)/fs:.2f} seconds)")
        
        if view_mode == "Doctor View":
            # Display clean ECG visualization and basic analysis for clinical use
            
            # Clean ECG visualization
            st.subheader("ECG Recording")
            fig = go.Figure()
            time = np.arange(len(signal)) / fs
            
            fig.add_trace(go.Scatter(
                x=time, 
                y=signal,
                mode='lines',
                name='ECG Signal'
            ))
            
            # Add ECG paper styling for clinical view
            fig.update_layout(
                title="ECG Recording",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude (mV)",
                height=400,
                paper_bgcolor='white',
                plot_bgcolor='white',
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
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic clinical metrics and analysis
            if medical_analysis_available:
                # Run analyses
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Heart Rate Analysis")
                    try:
                        # Use ECGMedicalAnalysis for clinical metrics
                        medical_analyzer = ECGMedicalAnalysis(fs=fs)
                        medical_report = medical_analyzer.generate_clinical_report(signal)
                        
                        if medical_report and 'heart_rate' in medical_report and 'heart_rate' in medical_report['heart_rate']:
                            hr = medical_report['heart_rate']['heart_rate']
                            st.metric("Heart Rate", f"{hr:.1f} BPM")
                            
                            # QRS duration if available
                            if medical_report['qrs_complex']['qrs_duration']['mean'] is not None:
                                qrs_duration = medical_report['qrs_complex']['qrs_duration']['mean']
                                st.metric("QRS Duration", f"{qrs_duration:.1f} ms")
                        else:
                            st.warning("Could not calculate heart rate metrics")
                    except Exception as e:
                        st.error(f"Error in heart rate analysis: {str(e)}")
                        st.warning("Try switching to Developer Mode for more detailed diagnostics")
                
                with col2:
                    st.subheader("Arrhythmia Analysis")
                    try:
                        # Use ECGArrhythmiaClassifier for AF detection
                        classifier = ECGArrhythmiaClassifier()
                        af_prob, af_metrics = classifier.detect_af(signal, sampling_rate=fs)
                        
                        # Display results
                        st.metric("AF Probability", f"{af_prob*100:.1f}%")
                        
                        # Classification result
                        if af_prob >= 0.7:
                            st.error("High probability of Atrial Fibrillation")
                        elif af_prob >= 0.3:
                            st.warning("Moderate probability of Atrial Fibrillation")
                        else:
                            st.success("Low probability of Atrial Fibrillation")
                            
                        # Basic metrics
                        st.metric("Mean Heart Rate", f"{af_metrics.get('mean_hr', 0):.1f} BPM")
                    except Exception as e:
                        st.error(f"Error in arrhythmia analysis: {str(e)}")
                        st.warning("Try switching to Developer Mode for more detailed diagnostics")
            else:
                st.warning("Medical analysis modules not available. Cannot perform clinical analysis.")
                
            # Option to show more detailed diagnostic info
            with st.expander("Show Additional Details"):
                if medical_analysis_available:
                    try:
                        # Show ECG with R-peaks
                        medical_analyzer = ECGMedicalAnalysis(fs=fs)
                        medical_report = medical_analyzer.generate_clinical_report(signal)
                        
                        if medical_report and 'heart_rate' in medical_report and 'r_peaks' in medical_report['heart_rate']:
                            st.subheader("ECG Signal with R-peaks")
                            r_peaks = medical_report['heart_rate']['r_peaks']
                            
                            fig = go.Figure()
                            time = np.arange(len(signal)) / fs
                            
                            fig.add_trace(go.Scatter(
                                x=time, 
                                y=signal,
                                mode='lines',
                                name='ECG Signal'
                            ))
                            
                            # Add R-peaks
                            if len(r_peaks) > 0:
                                peak_times = [time[p] for p in r_peaks if p < len(time)]
                                peak_amplitudes = [signal[p] for p in r_peaks if p < len(signal)]
                                
                                fig.add_trace(go.Scatter(
                                    x=peak_times,
                                    y=peak_amplitudes,
                                    mode='markers',
                                    marker=dict(color='red', size=10),
                                    name='R-peaks'
                                ))
                            
                            fig.update_layout(
                                height=400,
                                xaxis_title="Time (s)",
                                yaxis_title="Amplitude"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying detailed view: {str(e)}")
        
        else:  # Developer/Debug Mode
            if debug_section == "Signal Viewer":
                st.header("ECG Signal Viewer and Diagnostics")
                
                # Basic signal plot
                fig = go.Figure()
                time = np.arange(len(signal)) / fs
                
                fig.add_trace(go.Scatter(
                    x=time, 
                    y=signal,
                    mode='lines',
                    name='ECG Signal'
                ))
                
                fig.update_layout(
                    title="ECG Signal Overview",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Advanced diagnostics
                with st.expander("Show Signal Diagnostics"):
                    diag_fig = plot_signal_debug(signal, fs=fs)
                    st.plotly_chart(diag_fig, use_container_width=True)
                
                # Data preview
                if df is not None:
                    with st.expander("Show Data Preview"):
                        st.dataframe(df.head(100))
                
                # Allow downloading the data
                if df is not None:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Data as CSV",
                        data=csv,
                        file_name=f"ecg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            elif debug_section == "Processing Pipeline":
                st.header("ECG Signal Processing Pipeline")
                st.markdown("""
                This tool visualizes each step of the ECG signal processing pipeline to help identify 
                where issues might occur in the preprocessing of the signal.
                """)
                
                visualize_processing_steps(signal, fs=fs)
                
            elif debug_section == "R-peak Detection":
                st.header("R-peak Detection Comparison")
                st.markdown("""
                This tool compares different R-peak detection methods to help troubleshoot
                issues with heart rate calculation and other metrics that rely on accurate R-peak detection.
                """)
                
                test_r_peak_detection(signal, fs=fs)
                
            elif debug_section == "AF Detection Test":
                st.header("Atrial Fibrillation Detection Test")
                st.markdown("""
                This tool tests the AF detection algorithm on the current signal to help
                diagnose issues with AF probability calculation.
                """)
                
                if medical_analysis_available:
                    try:
                        # Use the ECGArrhythmiaClassifier to detect AF
                        classifier = ECGArrhythmiaClassifier()
                        af_prob, af_metrics = classifier.detect_af(signal, sampling_rate=fs)
                        
                        # Display results
                        st.markdown("### AF Detection Results")
                        
                        # Create gauge chart for AF probability
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = af_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "AF Probability (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "orange"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': af_prob * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display metrics
                        st.markdown("### AF Detection Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean Heart Rate", f"{af_metrics.get('mean_hr', 0):.1f} BPM")
                        col2.metric("SDNN", f"{af_metrics.get('sdnn', 0):.4f}")
                        col3.metric("RMSSD", f"{af_metrics.get('rmssd', 0):.4f}")
                        col4.metric("pNN50", f"{af_metrics.get('pnn50', 0)*100:.1f}%")
                        
                        # Show how the probability was calculated
                        st.markdown("### AF Probability Calculation Details")
                        
                        # Get the internal calculations if possible
                        st.code(f"""
Probability Calculation Factors:
- Irregularity metric: {af_metrics.get('irregularity', 0):.4f}
- RMSSD: {af_metrics.get('rmssd', 0):.4f}
- pNN50: {af_metrics.get('pnn50', 0):.4f}
- Mean HR: {af_metrics.get('mean_hr', 0):.1f} BPM

Final AF Probability: {af_prob:.4f} ({af_prob*100:.1f}%)
                        """)
                        
                        # Show R-peaks used for AF detection
                        st.markdown("### R-peaks Used in AF Detection")
                        
                        try:
                            # Try to get R-peaks using similar method as in detect_af
                            signal_preprocessed = classifier.preprocess_signal_for_rpeaks(signal, fs)
                            _, r_peaks = sp_signal.find_peaks(signal_preprocessed, distance=int(fs*0.3))
                            
                            # Plot signal with R-peaks
                            fig = go.Figure()
                            time = np.arange(len(signal)) / fs
                            
                            fig.add_trace(go.Scatter(
                                x=time, 
                                y=signal,
                                mode='lines',
                                name='ECG Signal'
                            ))
                            
                            # Add R-peaks
                            if len(r_peaks) > 0:
                                peak_times = [time[p] for p in r_peaks if p < len(time)]
                                peak_amplitudes = [signal[p] for p in r_peaks if p < len(signal)]
                                
                                fig.add_trace(go.Scatter(
                                    x=peak_times,
                                    y=peak_amplitudes,
                                    mode='markers',
                                    marker=dict(color='red', size=10),
                                    name='R-peaks'
                                ))
                            
                            fig.update_layout(
                                title=f"ECG Signal with Detected R-peaks ({len(r_peaks)} peaks)",
                                xaxis_title="Time (s)",
                                yaxis_title="Amplitude",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display RR intervals if applicable
                            if len(r_peaks) > 1:
                                rr_intervals = np.diff(r_peaks) / fs
                                
                                # Plot RR intervals
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=np.arange(len(rr_intervals)),
                                    y=rr_intervals,
                                    mode='lines+markers',
                                    name='RR Intervals'
                                ))
                                
                                fig.update_layout(
                                    title="RR Intervals",
                                    xaxis_title="Interval Number",
                                    yaxis_title="Interval Duration (s)",
                                    height=300
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Plot Poincaré plot of RR intervals
                                if len(rr_intervals) > 2:
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=rr_intervals[:-1],
                                        y=rr_intervals[1:],
                                        mode='markers',
                                        marker=dict(
                                            color='blue',
                                            size=8
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
                                        line=dict(dash='dash', color='gray'),
                                        name='Identity Line'
                                    ))
                                    
                                    fig.update_layout(
                                        title="Poincaré Plot (RRn+1 vs RRn)",
                                        xaxis_title="RR Interval n (s)",
                                        yaxis_title="RR Interval n+1 (s)",
                                        height=400,
                                        xaxis=dict(range=[min_rr*0.9, max_rr*1.1]),
                                        yaxis=dict(range=[min_rr*0.9, max_rr*1.1])
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error visualizing R-peaks: {str(e)}")
                            
                    except Exception as e:
                        st.error(f"Error in AF detection: {str(e)}")
                else:
                    st.warning("AF detection requires the ECGArrhythmiaClassifier module which is not available.")
                
            elif debug_section == "Info":
                st.header("About ECG Diagnostics")
                st.markdown("""
                This diagnostic tool is designed to help troubleshoot issues with ECG signal visualization and processing.
                It provides insight into each step of the signal processing pipeline, and helps identify where problems
                might occur in the preprocessing, feature extraction, or classification processes.
                
                ### Key Features:
                
                1. **Signal Viewer**: Visualize the raw ECG signal and basic diagnostic information
                2. **Processing Pipeline**: Step through each preprocessing stage to spot issues
                3. **R-peak Detection**: Compare different R-peak detection methods
                4. **AF Detection Test**: Test the atrial fibrillation detection algorithm
                
                ### Troubleshooting Tips:
                
                - If no signal appears in the visualization, check your input data format
                - Ensure the sampling rate is correctly specified
                - For R-peak detection issues, check the signal quality and amplitude
                - For AF detection problems, verify that enough R-peaks are being detected
                
                ### Data Format:
                
                The tool accepts the following data formats:
                - CSV files with 'time' and 'signal' columns
                - CSV files where the first column is assumed to be time and the second is signal
                - Plain text files with one signal value per line
                - Synthetic test signal generation
                - Sample segments from EDF files (if available)
                """)
    else:
        st.warning("Please upload ECG data or generate a synthetic signal to begin analysis.")

if __name__ == "__main__":
    main() 