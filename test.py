import streamlit as st
import numpy as np
import pandas as pd
import mne
import neurokit2 as nk
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

# Set page config
st.set_page_config(page_title="ECG EDF Analyzer", page_icon="❤️", layout="wide")

# Custom styling for ECG-like appearance
plt.style.use('dark_background')

def validate_and_repair_ecg_data(data, fs=128):
    """
    Validate ECG data and attempt to repair common issues
    Returns repaired data and a list of issues found
    """
    issues = []
    repaired_data = data.copy()
    
    # Check for identical channels
    identical_channels = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            if np.allclose(data[i][:1000], data[j][:1000], rtol=1e-5, atol=1e-8):
                identical_channels.append((i, j))
    
    if identical_channels:
        issues.append("Identical channels detected")
        
        # If all channels are identical, try to create synthetic variations
        if len(identical_channels) == (data.shape[0] * (data.shape[0] - 1)) // 2:
            issues.append("All channels are identical - creating synthetic variations")
            
            # Keep first channel as is
            # Add slight variations to other channels
            for i in range(1, data.shape[0]):
                # Add progressively more variation to each channel
                noise_level = 0.1 * i
                repaired_data[i] = data[i] + noise_level * np.random.normal(0, np.std(data[i]) or 0.01, data[i].shape)
    
    # Check for consecutive identical values
    for i in range(data.shape[0]):
        consecutive_identical = 0
        max_consecutive = 0
        for j in range(1, min(10000, data.shape[1])):
            if data[i][j] == data[i][j-1]:
                consecutive_identical += 1
            else:
                max_consecutive = max(max_consecutive, consecutive_identical)
                consecutive_identical = 0
        
        if max_consecutive > 10:
            issues.append(f"Channel {i} has {max_consecutive} consecutive identical values")
            
            # Repair consecutive identical values by adding small variations
            for j in range(1, data.shape[1]):
                if repaired_data[i][j] == repaired_data[i][j-1]:
                    # Add tiny random variation
                    repaired_data[i][j] += np.random.normal(0, 0.00001)
    
    # Check for very small data range
    data_range = np.ptp(data)
    if 0 < data_range < 0.01:
        issues.append(f"Very small data range: {data_range:.8f}")
        
        # Normalize and rescale to more typical ECG range
        for i in range(data.shape[0]):
            channel_min = np.min(repaired_data[i])
            channel_max = np.max(repaired_data[i])
            if channel_max > channel_min:
                # Normalize to 0-1
                repaired_data[i] = (repaired_data[i] - channel_min) / (channel_max - channel_min)
                # Scale to typical ECG range (-1 to 1 mV)
                repaired_data[i] = repaired_data[i] * 2 - 1
    
    return repaired_data, issues

def create_ecg_style_plot(fig, ax, x, y, title, xlabel="Time (s)", ylabel="Amplitude (mV)"):
    """Create a plot that looks like a real ECG monitor"""
    # Plot with green line on black background
    ax.plot(x, y, color='#00FF00', linewidth=1.5)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Add grid lines like an ECG monitor
    ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
    
    # Add title and labels
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    
    # Style the ticks
    ax.tick_params(colors='white')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#444444', linestyle='-', alpha=0.5)
    
    return fig, ax

def main():
    st.title("❤️ ECG EDF Analyzer")
    st.write("Upload an EDF file to analyze ECG data.")
    
    uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(tmp_path, preload=True)
            fs = int(raw.info['sfreq'])
            data = raw.get_data()
            channel_names = raw.ch_names
            times = np.arange(data.shape[1]) / fs
            
            # Validate and repair ECG data
            repaired_data, issues = validate_and_repair_ecg_data(data, fs)
            
            if issues:
                st.warning("Data issues detected:")
                for issue in issues:
                    st.write(issue)
            
            data = repaired_data
            
            # Display file info
            st.success(f"File loaded successfully. Found {len(channel_names)} channels, sampling rate: {fs} Hz")
            
            # Display overall data statistics
            st.write(f"Data shape: {data.shape}, Time points: {data.shape[1]}")
            
            # Check for NaN, Inf, or all zeros
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
            all_zeros = np.allclose(data, 0, atol=1e-10)
            
            st.write(f"Data contains NaN: {has_nan}, Inf: {has_inf}, All zeros: {all_zeros}")
            
            # Display overall data range
            st.write(f"Overall data range - Min: {np.min(data):.8f}, Max: {np.max(data):.8f}, Mean: {np.mean(data):.8f}")
            
            # Check if channels are identical
            identical_channels = []
            for i in range(len(channel_names)):
                for j in range(i+1, len(channel_names)):
                    if np.allclose(data[i][:1000], data[j][:1000], rtol=1e-5, atol=1e-8):
                        identical_channels.append((channel_names[i], channel_names[j]))
            
            if identical_channels:
                st.warning("⚠️ Warning: The following channels appear to be identical:")
                for ch1, ch2 in identical_channels:
                    st.write(f"- {ch1} and {ch2}")
                st.write("This may indicate that the file contains duplicate data or synthetic data.")
            
            # Check for variability in the data (real ECG data should have variability)
            for i, name in enumerate(channel_names):
                # Calculate standard deviation and coefficient of variation
                std_dev = np.std(data[i][:1000])
                mean_val = np.mean(data[i][:1000])
                if mean_val != 0:
                    cv = std_dev / abs(mean_val)
                else:
                    cv = 0
                
                # Check for repeating patterns
                autocorr = np.correlate(data[i][:1000] - np.mean(data[i][:1000]), 
                                       data[i][:1000] - np.mean(data[i][:1000]), 
                                       mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                # Real ECG data typically has a coefficient of variation > 0.1
                # and shows clear peaks in autocorrelation at regular intervals
                if cv < 0.01:
                    st.warning(f"⚠️ Channel {name} has very low variability (CV={cv:.4f}), which is unusual for real ECG data.")
                
                # Check for constant values or repeating patterns
                consecutive_equal = 0
                max_consecutive = 0
                for j in range(1, min(1000, len(data[i]))):
                    if data[i][j] == data[i][j-1]:
                        consecutive_equal += 1
                    else:
                        max_consecutive = max(max_consecutive, consecutive_equal)
                        consecutive_equal = 0
                
                if max_consecutive > 10:
                    st.warning(f"⚠️ Channel {name} has {max_consecutive} consecutive identical values, which is unusual for real ECG data.")
            
            # Channel preview
            st.subheader("📊 Channel Preview")
            
            # Let user select which channels to preview
            preview_samples = 640  # Show first 5 seconds at 128 Hz
            
            # Options for channel selection
            channel_options = ["All channels"] + [f"First {n} channels" for n in [1, 3, 5, 10]] + ["Select specific channels"]
            channel_display_option = st.radio("Display options:", channel_options)
            
            if channel_display_option == "All channels":
                channels_to_show = channel_names
            elif channel_display_option == "Select specific channels":
                channels_to_show = st.multiselect("Select channels to display:", channel_names, default=channel_names[:3])
            elif channel_display_option.startswith("First"):
                n = int(channel_display_option.split()[1])
                channels_to_show = channel_names[:min(n, len(channel_names))]
            
            for name in channels_to_show:
                idx = channel_names.index(name)
                segment = data[idx][:preview_samples]
                
                # Calculate range to detect flat signals
                signal_range = np.ptp(segment)
                
                # Diagnostic info for this channel
                st.write(f"Channel {name} - Shape: {segment.shape}, Range: {signal_range:.8f}")
                st.write(f"First 5 values: {segment[:5]}")
                
                # Skip if data is all zeros or NaN
                if np.isnan(segment).all() or np.allclose(segment, 0, atol=1e-10):
                    st.warning(f"Channel {name} contains no usable data (all NaN or zeros)")
                    continue
                
                # Scale the data to make it visible
                segment_scaled = segment * 1000  # Scale by 1000 to make small values visible
                
                # Debug information about the scaled data
                st.write(f"Scaled data - Min: {np.min(segment_scaled):.4f}, Max: {np.max(segment_scaled):.4f}, Range: {np.ptp(segment_scaled):.4f}")
                
                # Try different scaling methods if range is still small
                if np.ptp(segment_scaled) < 2.0:
                    st.info("Range is still small after scaling, trying normalization...")
                    # Normalize to 0-1 range
                    segment_norm = (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-10)
                    # Scale to reasonable ECG range
                    segment_scaled = segment_norm * 2 - 1  # Scale to -1 to 1 range
                    st.write(f"Normalized data - Min: {np.min(segment_scaled):.4f}, Max: {np.max(segment_scaled):.4f}, Range: {np.ptp(segment_scaled):.4f}")
                
                # Add small random variations if data appears too uniform
                if len(np.unique(segment_scaled[:20])) < 5:
                    st.info("Data appears uniform, adding small variations for visualization...")
                    # Add small variations to make the signal more visible
                    segment_scaled = segment_scaled + np.random.normal(0, 0.05, len(segment_scaled))
                
                # Create ECG-style plot with Matplotlib
                fig, ax = plt.subplots(figsize=(10, 3))
                fig, ax = create_ecg_style_plot(fig, ax, times[:preview_samples], segment_scaled, 
                                               f"ECG: {name} (Range: {signal_range:.6f})")
                st.pyplot(fig)
            
            # Let user select channel for detailed analysis
            st.subheader("🔍 Detailed Channel Analysis")
            st.write("Select signal channel for analysis:")
            selected_channel = st.selectbox("", channel_names)
            channel_index = channel_names.index(selected_channel)
            
            # Extract signal and time
            signal = data[channel_index]
            time = times
            
            st.subheader(f"Signal: {selected_channel} ({len(signal)} samples, {fs} Hz)")
            st.write(f"Min: {np.min(signal):.6f}, Max: {np.max(signal):.6f}, Mean: {np.mean(signal):.6f}, Range: {np.ptp(signal):.6f}")
            
            # Check if signal contains any non-zero values
            non_zero_count = np.count_nonzero(signal)
            st.write(f"Non-zero values: {non_zero_count} out of {len(signal)} ({non_zero_count/len(signal)*100:.2f}%)")
            
            # Check if signal contains any NaN values
            nan_count = np.isnan(signal).sum()
            if nan_count > 0:
                st.warning(f"Warning: Signal contains {nan_count} NaN values")
                # Replace NaN with zeros for processing
                signal = np.nan_to_num(signal)
            
            # Use a much smaller threshold for flatness detection
            if np.ptp(signal) < 0.0001:
                st.warning("⚠️ The selected channel appears to be flat or contains no useful ECG data.")
                
                # Try to visualize anyway with a magnified view
                st.info("Showing magnified view of the signal (may appear as noise)")
                
                # Add a small amount of noise if signal is completely flat
                if np.ptp(signal) == 0:
                    signal = signal + np.random.normal(0, 1e-6, len(signal))
                
                # Scale by 1000 for visibility
                signal_scaled = signal * 1000
                
                # Debug information
                st.write(f"Scaled data - Min: {np.min(signal_scaled):.4f}, Max: {np.max(signal_scaled):.4f}, Range: {np.ptp(signal_scaled):.4f}")
                
                # Create ECG-style plot
                fig, ax = plt.subplots(figsize=(12, 5))
                fig, ax = create_ecg_style_plot(fig, ax, time[:1000], signal_scaled[:1000], 
                                              "Raw ECG Signal (Magnified View)")
                st.pyplot(fig)
                
            else:
                # Scale by 1000 for visibility
                signal_scaled = signal * 1000
                
                # Debug information
                st.write(f"Scaled data - Min: {np.min(signal_scaled):.4f}, Max: {np.max(signal_scaled):.4f}, Range: {np.ptp(signal_scaled):.4f}")
                
                # Try different scaling methods if range is still small
                if np.ptp(signal_scaled) < 2.0:
                    st.info("Range is still small after scaling, trying normalization...")
                    # Normalize to 0-1 range
                    signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
                    # Scale to reasonable ECG range
                    signal_scaled = signal_norm * 2 - 1  # Scale to -1 to 1 range
                    st.write(f"Normalized data - Min: {np.min(signal_scaled):.4f}, Max: {np.max(signal_scaled):.4f}, Range: {np.ptp(signal_scaled):.4f}")
                
                # Add small random variations if data appears too uniform
                if len(np.unique(signal_scaled[:20])) < 5:
                    st.info("Data appears uniform, adding small variations for visualization...")
                    # Add small variations to make the signal more visible
                    signal_scaled = signal_scaled + np.random.normal(0, 0.05, len(signal_scaled))
                
                # Create ECG-style plot
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # Only show a portion of the data for better visualization
                display_length = min(5000, len(signal_scaled))
                
                # Create ECG-style plot with a subset of data for better performance
                fig, ax = create_ecg_style_plot(fig, ax, time[:display_length], signal_scaled[:display_length], 
                                              "Raw ECG Signal")
                
                st.pyplot(fig)
                
                # Process ECG with NeuroKit2
                st.subheader("🔬 ECG Feature Extraction")
                # Use original unscaled signal for processing
                ecg_cleaned = nk.ecg_clean(signal, sampling_rate=fs)
                signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)
                
                # Plot processed ECG
                st.write("### Processed ECG Signal")
                
                # Create ECG-style plot for processed signal
                processed_scaled = signals["ECG_Clean"] * 1000
                fig, ax = plt.subplots(figsize=(12, 5))
                fig, ax = create_ecg_style_plot(fig, ax, signals.index[:1000]/fs, processed_scaled[:1000], 
                                              "Processed ECG Signal")
                
                # Mark R-peaks if available
                if "ECG_R_Peaks" in signals.columns:
                    r_peaks = signals["ECG_R_Peaks"][:1000]
                    r_peak_times = np.where(r_peaks == 1)[0] / fs
                    r_peak_values = [processed_scaled[int(peak * fs)] for peak in r_peak_times if int(peak * fs) < 1000]
                    r_peak_times = [time for time in r_peak_times if time * fs < 1000]
                    if len(r_peak_times) > 0:
                        ax.scatter(r_peak_times, r_peak_values, color='red', s=50, zorder=3, label='R-peaks')
                        ax.legend(loc='upper right')
                
                st.pyplot(fig)
                
                # Display heart rate and other metrics
                st.write("### Heart Rate Analysis")
                
                if "ECG_Rate" in signals.columns:
                    mean_hr = np.mean(signals["ECG_Rate"].dropna())
                    st.metric("Average Heart Rate", f"{mean_hr:.1f} BPM")
                    
                    # Create ECG-style plot for heart rate
                    fig, ax = plt.subplots(figsize=(12, 4))
                    valid_hr = ~np.isnan(signals["ECG_Rate"])
                    hr_times = signals.index[valid_hr]/fs
                    hr_values = signals["ECG_Rate"][valid_hr]
                    
                    ax.plot(hr_times[:1000], hr_values[:1000], color='#00FF00', linewidth=1.5)
                    ax.set_facecolor('black')
                    fig.patch.set_facecolor('black')
                    ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
                    ax.set_title("Heart Rate Variability", color='white', fontsize=14)
                    ax.set_xlabel("Time (s)", color='white')
                    ax.set_ylabel("Heart Rate (BPM)", color='white')
                    ax.tick_params(colors='white')
                    
                    st.pyplot(fig)
                
                # Display HRV metrics
                if "HRV" in info:
                    st.write("### Heart Rate Variability Metrics")
                    hrv_df = pd.DataFrame(info["HRV"]).T
                    st.dataframe(hrv_df)
                
                # Generate PDF report option
                st.subheader("📄 Generate Report")
                if st.button("Generate PDF Report"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Add title
                    pdf.cell(200, 10, txt=f"ECG Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                    pdf.ln(10)
                    
                    # Add file info
                    pdf.cell(200, 10, txt=f"File: {uploaded_file.name}", ln=True)
                    pdf.cell(200, 10, txt=f"Channel: {selected_channel}", ln=True)
                    pdf.cell(200, 10, txt=f"Sampling Rate: {fs} Hz", ln=True)
                    pdf.ln(10)
                    
                    # Add heart rate info if available
                    if "ECG_Rate" in signals.columns:
                        pdf.cell(200, 10, txt=f"Average Heart Rate: {mean_hr:.1f} BPM", ln=True)
                    
                    # Save PDF to temp file and provide download link
                    pdf_path = os.path.join(tempfile.gettempdir(), "ecg_report.pdf")
                    pdf.output(pdf_path)
                    
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="ecg_report.pdf",
                        mime="application/pdf"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()
