import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import tempfile
from datetime import datetime, timedelta
import hashlib

# Import project modules with graceful error handling
try:
    from ecg_holter_analysis import HolterAnalyzer
    from ecg_arrhythmia_classification import ECGArrhythmiaClassifier
    from ecg_medical_analysis import ECGMedicalAnalysis
    modules_available = True
except ImportError:
    st.warning("Could not import ECG analysis modules. Some functionality may be limited.")
    modules_available = False

def diagnose_signal(signal, fs=200):
    """
    Diagnose common signal issues that might cause visualization problems
    
    Args:
        signal: numpy array containing the signal
        fs: sampling frequency (Hz)
        
    Returns:
        dict with diagnostic results
    """
    diagnostics = {}
    
    # Check for empty or None signal
    if signal is None:
        diagnostics["signal_exists"] = False
        return diagnostics
    
    diagnostics["signal_exists"] = True
    diagnostics["signal_length"] = len(signal)
    diagnostics["signal_duration"] = f"{len(signal)/fs:.2f} seconds"
    
    # Check for NaN values
    nan_count = np.isnan(signal).sum()
    diagnostics["contains_nan"] = nan_count > 0
    diagnostics["nan_count"] = nan_count
    
    # Check for Inf values
    inf_count = np.isinf(signal).sum()
    diagnostics["contains_inf"] = inf_count > 0
    diagnostics["inf_count"] = inf_count
    
    # Check for zero variance sections
    if len(signal) > 10:
        rolling_std = pd.Series(signal).rolling(window=int(fs/2)).std()
        zero_var_sections = (rolling_std < 1e-6).sum()
        diagnostics["zero_variance_sections"] = zero_var_sections
        diagnostics["zero_variance_percent"] = f"{zero_var_sections / len(rolling_std) * 100:.2f}%"
    else:
        diagnostics["zero_variance_sections"] = 0
        diagnostics["zero_variance_percent"] = "0.00%"
    
    # Check signal range
    diagnostics["min_value"] = np.min(signal)
    diagnostics["max_value"] = np.max(signal)
    diagnostics["signal_range"] = np.max(signal) - np.min(signal)
    
    # Check if signal is too small
    diagnostics["signal_too_small"] = diagnostics["signal_range"] < 0.01
    
    # Check if signal is mostly flatlined
    if len(signal) > 10:
        diff_signal = np.abs(np.diff(signal))
        flat_threshold = 1e-6
        flatline_percent = np.sum(diff_signal < flat_threshold) / len(diff_signal) * 100
        diagnostics["flatline_percent"] = f"{flatline_percent:.2f}%"
        diagnostics["mostly_flatlined"] = flatline_percent > 80
    else:
        diagnostics["flatline_percent"] = "0.00%"
        diagnostics["mostly_flatlined"] = False
    
    return diagnostics

def plot_signal_with_diagnostics(df, title="Signal Diagnostics"):
    """
    Plot signal with diagnostic information
    
    Args:
        df: DataFrame with time and signal columns
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if df is None or len(df) == 0:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Add signal trace
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['signal'],
        mode='lines',
        name='ECG Signal',
        line=dict(
            color='rgb(0, 0, 255)',
            width=2,
        ),
        opacity=0.8
    ))
    
    # Add zero line for reference
    fig.add_shape(
        type="line",
        x0=min(df['time']),
        y0=0,
        x1=max(df['time']),
        y1=0,
        line=dict(color="rgba(0, 0, 0, 0.5)", width=1, dash="dash"),
    )
    
    # Check for flat sections
    if len(df) > 10:
        rolling_std = pd.Series(df['signal']).rolling(window=5).std()
        flat_sections = rolling_std < 1e-6
        
        # Mark flat sections with a different color
        if flat_sections.sum() > 0:
            # Convert boolean mask to regions
            flat_regions = []
            in_flat = False
            start_idx = 0
            
            for i, is_flat in enumerate(flat_sections):
                if is_flat and not in_flat:
                    start_idx = i
                    in_flat = True
                elif not is_flat and in_flat:
                    # Add range of flat section
                    if i - start_idx > 2:  # Only mark if flat section is significant
                        flat_regions.append((start_idx, i))
                    in_flat = False
            
            # Handle case where series ends during a flat section
            if in_flat and len(df) - start_idx > 2:
                flat_regions.append((start_idx, len(df) - 1))
            
            # Add flat regions to plot
            for start, end in flat_regions:
                fig.add_trace(go.Scatter(
                    x=df['time'].iloc[start:end],
                    y=df['signal'].iloc[start:end],
                    mode='lines',
                    name='Flat Section',
                    line=dict(color='rgba(255, 0, 0, 0.8)', width=3),
                ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        hovermode="closest",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Ensure X and Y ranges are appropriate
    signal_min = df['signal'].min()
    signal_max = df['signal'].max()
    signal_range = signal_max - signal_min
    y_min = signal_min - signal_range * 0.1 if signal_range > 0 else signal_min - 0.1
    y_max = signal_max + signal_range * 0.1 if signal_range > 0 else signal_max + 0.1
    
    fig.update_layout(
        xaxis=dict(range=[df['time'].min(), df['time'].max()]),
        yaxis=dict(range=[y_min, y_max])
    )
    
    return fig

def visualize_improved_signal(df, diagnostics):
    """
    Create an improved visualization of signal based on diagnostics
    
    Args:
        df: DataFrame with time and signal columns
        diagnostics: Dictionary with diagnostic results
        
    Returns:
        Plotly figure with improved visualization
    """
    if df is None or len(df) == 0:
        return go.Figure()
    
    # Copy DataFrame to avoid modifying original
    df_improved = df.copy()
    signal_improved = df_improved['signal'].values
    
    # Replace NaN values with interpolated values
    if diagnostics["contains_nan"]:
        df_improved['signal'] = df_improved['signal'].interpolate(method='linear', limit_direction='both')
    
    # Replace Inf values
    if diagnostics["contains_inf"]:
        inf_mask = np.isinf(signal_improved)
        if np.any(inf_mask):
            # Replace Inf with interpolated values or with min/max of non-Inf values
            non_inf_signal = signal_improved[~inf_mask]
            if len(non_inf_signal) > 0:
                min_val, max_val = np.min(non_inf_signal), np.max(non_inf_signal)
                # Replace +Inf with max, -Inf with min
                signal_improved[np.isposinf(signal_improved)] = max_val
                signal_improved[np.isneginf(signal_improved)] = min_val
                df_improved['signal'] = signal_improved
    
    # Amplify signal if too small
    if diagnostics["signal_too_small"]:
        # Scale signal to have a range of approximately 1.0
        signal_range = diagnostics["max_value"] - diagnostics["min_value"]
        if signal_range > 0:
            scale_factor = 1.0 / signal_range
            df_improved['signal'] = (df_improved['signal'] - np.mean(df_improved['signal'])) * scale_factor
    
    # Create figure
    fig = go.Figure()
    
    # Add original signal trace
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['signal'],
        mode='lines',
        name='Original Signal',
        line=dict(color='rgba(0, 0, 255, 0.3)', width=1),
        opacity=0.5
    ))
    
    # Add improved signal trace
    fig.add_trace(go.Scatter(
        x=df_improved['time'],
        y=df_improved['signal'],
        mode='lines',
        name='Improved Signal',
        line=dict(color='rgb(255, 0, 0)', width=2),
    ))
    
    # Update layout
    fig.update_layout(
        title="Improved Signal Visualization",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        hovermode="closest",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Ensure X and Y ranges are appropriate
    improved_min = df_improved['signal'].min()
    improved_max = df_improved['signal'].max()
    improved_range = improved_max - improved_min
    y_min = improved_min - improved_range * 0.1 if improved_range > 0 else improved_min - 0.1
    y_max = improved_max + improved_range * 0.1 if improved_range > 0 else improved_max + 0.1
    
    fig.update_layout(
        xaxis=dict(range=[df['time'].min(), df['time'].max()]),
        yaxis=dict(range=[y_min, y_max])
    )
    
    return fig, df_improved

def main():
    st.set_page_config(page_title="ECG Visualization Diagnostics", page_icon="❤️", layout="wide")
    
    st.title("ECG Visualization Diagnostics")
    st.markdown("""
    This tool helps diagnose and fix issues with ECG signal visualization.
    Upload an EDF file or use test data to diagnose why your ECG might not be displaying correctly.
    """)
    
    # Sidebar for controls
    st.sidebar.title("Controls")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Upload EDF File", "Use Test Data"]
    )
    
    # Load data
    df = None
    signal = None
    fs = 200  # Default sampling frequency
    
    if data_source == "Upload EDF File":
        uploaded_file = st.sidebar.file_uploader("Upload EDF File", type=["edf"])
        
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
                        fs = holter_analyzer.fs
                        
                        # Get segment
                        segment_start = st.sidebar.slider(
                            "Start time (minutes from beginning)",
                            min_value=0,
                            max_value=int(holter_analyzer.duration_hours * 60) - 1,
                            value=0,
                            step=1
                        )
                        
                        segment_duration = st.sidebar.slider(
                            "Segment duration (seconds)",
                            min_value=10,
                            max_value=300,
                            value=60,
                            step=10
                        )
                        
                        # Get the segment
                        df = holter_analyzer.get_segment(segment_start, segment_duration)
                        if df is not None:
                            signal = df['signal'].values
                    else:
                        st.error("Failed to load EDF file. The file may not be in the correct format.")
                else:
                    st.error("Unable to load EDF files because ECG modules are not available.")
                
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            except Exception as e:
                st.error(f"Error processing EDF file: {str(e)}")
    else:  # Use Test Data
        test_data_type = st.sidebar.selectbox(
            "Select Test Data Type",
            ["Normal ECG", "Flat ECG", "Noisy ECG", "ECG with NaN/Inf", "Very Small Amplitude", "AF Simulation"]
        )
        
        # Generate test data
        duration = st.sidebar.slider("Duration (seconds)", 1, 60, 10)
        fs = st.sidebar.slider("Sampling Frequency (Hz)", 100, 1000, 200)
        time = np.arange(0, duration, 1/fs)
        
        if test_data_type == "Normal ECG":
            # Simulate normal ECG pattern
            signal = []
            for t in time:
                # Create cardiac cycle (simplified)
                t_mod = t % 0.8  # 75 BPM
                if t_mod < 0.05:
                    val = -0.1
                elif t_mod < 0.1:
                    val = 0.05
                elif t_mod < 0.15:
                    val = 1.0  # R peak
                elif t_mod < 0.2:
                    val = -0.3
                elif t_mod < 0.3:
                    val = 0.2  # T wave
                else:
                    val = 0
                
                signal.append(val)
            
            signal = np.array(signal)
            
        elif test_data_type == "Flat ECG":
            # Create flatlined ECG
            signal = np.zeros(len(time))
            
        elif test_data_type == "Noisy ECG":
            # Simulate normal ECG with noise
            signal = []
            for t in time:
                # Create cardiac cycle (simplified)
                t_mod = t % 0.8  # 75 BPM
                if t_mod < 0.05:
                    val = -0.1
                elif t_mod < 0.1:
                    val = 0.05
                elif t_mod < 0.15:
                    val = 1.0  # R peak
                elif t_mod < 0.2:
                    val = -0.3
                elif t_mod < 0.3:
                    val = 0.2  # T wave
                else:
                    val = 0
                
                # Add noise
                val += np.random.normal(0, 0.2)
                signal.append(val)
            
            signal = np.array(signal)
            
        elif test_data_type == "ECG with NaN/Inf":
            # Simulate ECG with NaN and Inf values
            signal = []
            for t in time:
                # Create cardiac cycle (simplified)
                t_mod = t % 0.8  # 75 BPM
                if t_mod < 0.05:
                    val = -0.1
                elif t_mod < 0.1:
                    val = 0.05
                elif t_mod < 0.15:
                    val = 1.0  # R peak
                elif t_mod < 0.2:
                    val = -0.3
                elif t_mod < 0.3:
                    val = 0.2  # T wave
                else:
                    val = 0
                
                signal.append(val)
            
            signal = np.array(signal)
            
            # Add NaN and Inf values at specific positions
            nan_positions = np.random.choice(len(signal), size=int(len(signal)*0.05), replace=False)
            inf_positions = np.random.choice(len(signal), size=int(len(signal)*0.02), replace=False)
            
            signal[nan_positions] = np.nan
            signal[inf_positions] = np.inf
            
        elif test_data_type == "Very Small Amplitude":
            # Simulate ECG with very small amplitude
            signal = []
            for t in time:
                # Create cardiac cycle (simplified) but with very small amplitude
                t_mod = t % 0.8  # 75 BPM
                if t_mod < 0.05:
                    val = -0.0001
                elif t_mod < 0.1:
                    val = 0.00005
                elif t_mod < 0.15:
                    val = 0.001  # R peak
                elif t_mod < 0.2:
                    val = -0.0003
                elif t_mod < 0.3:
                    val = 0.0002  # T wave
                else:
                    val = 0
                
                signal.append(val)
            
            signal = np.array(signal)
            
        elif test_data_type == "AF Simulation":
            # Simulate AF with irregular RR intervals
            signal = []
            t_next_r = 0
            for t in time:
                if t >= t_next_r:
                    # Irregular R-R interval (typical for AF)
                    rr_interval = np.random.normal(0.6, 0.2)  # Mean HR 100, but highly variable
                    rr_interval = max(0.3, min(1.2, rr_interval))  # Limit to reasonable values
                    t_next_r = t + rr_interval
                    
                    # R peak
                    val = 1.0
                elif t >= t_next_r - 0.05:
                    val = 0.05  # Q wave
                elif t >= t_next_r - 0.1:
                    val = -0.1  # P wave absent in AF
                elif t <= t_next_r - 0.75 and t >= t_next_r - 0.65:
                    val = 0.2  # T wave
                elif t <= t_next_r - 0.7 and t >= t_next_r - 0.75:
                    val = -0.3
                else:
                    val = 0
                
                # Add fibrillatory waves (small irregular oscillations)
                val += 0.05 * np.sin(t * 40) + 0.03 * np.sin(t * 73) + 0.02 * np.sin(t * 57)
                
                signal.append(val)
            
            signal = np.array(signal)
        
        df = pd.DataFrame({'time': time, 'signal': signal})
    
    # Main content area
    if df is not None and signal is not None:
        # Run diagnostics
        diagnostics = diagnose_signal(signal, fs)
        
        # Display diagnostics
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.plotly_chart(plot_signal_with_diagnostics(df, "Original Signal"), use_container_width=True)
        
        with col2:
            st.subheader("Signal Diagnostics")
            for key, value in diagnostics.items():
                if key in ["signal_exists", "contains_nan", "contains_inf", "signal_too_small", "mostly_flatlined"]:
                    status = "🟢" if not (value if key != "signal_exists" else not value) else "🔴"
                    st.write(f"{status} {key.replace('_', ' ').title()}: {value}")
                elif key in ["flatline_percent", "zero_variance_percent"]:
                    percent_value = float(value.strip('%'))
                    status = "🟢" if percent_value < 10 else "🟠" if percent_value < 50 else "🔴"
                    st.write(f"{status} {key.replace('_', ' ').title()}: {value}")
            
            st.write("---")
            st.subheader("Signal Properties")
            st.write(f"Length: {diagnostics['signal_length']} samples")
            st.write(f"Duration: {diagnostics['signal_duration']}")
            st.write(f"Range: {diagnostics['signal_range']:.6f}")
            st.write(f"Min: {diagnostics['min_value']:.6f}")
            st.write(f"Max: {diagnostics['max_value']:.6f}")
        
        # Display improved signal if there are issues to fix
        if (diagnostics["contains_nan"] or diagnostics["contains_inf"] or 
            diagnostics["signal_too_small"] or diagnostics["mostly_flatlined"]):
            
            st.subheader("Improved Visualization")
            improved_fig, improved_df = visualize_improved_signal(df, diagnostics)
            st.plotly_chart(improved_fig, use_container_width=True)
            
            st.subheader("Recommendations")
            if diagnostics["contains_nan"]:
                st.markdown("""
                **Fix NaN Values**: Your signal contains NaN values which breaks visualization. 
                Consider using interpolation to fill these values:
                ```python
                df['signal'] = df['signal'].interpolate(method='linear', limit_direction='both')
                ```
                """)
            
            if diagnostics["contains_inf"]:
                st.markdown("""
                **Fix Infinite Values**: Your signal contains infinite values.
                Replace these with reasonable limits based on the rest of your signal:
                ```python
                # Replace infinites with signal limits
                inf_mask = np.isinf(df['signal'])
                non_inf_vals = df.loc[~inf_mask, 'signal']
                df.loc[np.isposinf(df['signal']), 'signal'] = non_inf_vals.max() if len(non_inf_vals) > 0 else 1.0
                df.loc[np.isneginf(df['signal']), 'signal'] = non_inf_vals.min() if len(non_inf_vals) > 0 else -1.0
                ```
                """)
            
            if diagnostics["signal_too_small"]:
                st.markdown("""
                **Amplify Signal**: Your signal amplitude is too small to be visible.
                Scale the signal to increase its visibility:
                ```python
                # Normalize and scale to visible range
                signal_range = df['signal'].max() - df['signal'].min()
                if signal_range > 0:
                    df['signal'] = (df['signal'] - df['signal'].mean()) * (1.0 / signal_range)
                ```
                """)
            
            if diagnostics["mostly_flatlined"]:
                st.markdown("""
                **Flatline Detection**: Your signal appears to be mostly flat.
                Check your data source or preprocessing steps:
                1. Verify the source of your ECG data
                2. Check your filters - they might be too aggressive
                3. Ensure your electrode connections are good
                """)
            
            # Option to download improved data
            csv = improved_df.to_csv(index=False)
            st.download_button(
                label="Download Improved Signal Data",
                data=csv,
                file_name=f"improved_ecg_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
            )
        
        # Add visualization with different plotting libraries for comparison
        st.subheader("Visualization Comparison")
        vis_type = st.radio(
            "Select Visualization Type",
            ["Plotly", "Matplotlib", "Streamlit Line Chart"],
            horizontal=True
        )
        
        if vis_type == "Plotly":
            # Plotly visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['signal'],
                mode='lines',
                name='ECG Signal',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Plotly Visualization",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude (mV)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif vis_type == "Matplotlib":
            # Matplotlib visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['time'], df['signal'])
            ax.set_title("Matplotlib Visualization")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude (mV)")
            ax.grid(True)
            st.pyplot(fig)
            
        else:  # Streamlit Line Chart
            # Streamlit native chart
            st.line_chart(
                df.rename(columns={'signal': 'ECG Signal'}).set_index('time'),
                height=400
            )
            st.caption("Streamlit Line Chart Visualization")

if __name__ == "__main__":
    main() 