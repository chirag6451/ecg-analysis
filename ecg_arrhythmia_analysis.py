import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import wfdb
import neurokit2 as nk
from scipy import signal
import pandas as pd

def analyze_ecg_edf(file_path, output_pdf="ecg_analysis_report.pdf"):
    """Analyze an ECG from an EDF file using MNE."""
    try:
        import mne
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Load the EDF file with MNE
        raw = mne.io.read_raw_edf(file_path, preload=True)
        print(f"Successfully loaded {file_path}")
        print(f"Available channels: {raw.ch_names}")
        
        # Choose the ECG channel - you may need to adjust this based on your file
        ecg_channel = raw.ch_names[0]  # Use first channel by default
        for ch in raw.ch_names:
            if 'ecg' in ch.lower() or 'ekg' in ch.lower():
                ecg_channel = ch
                break
        
        print(f"Using ECG channel: {ecg_channel}")
        
        # Extract data
        data, times = raw.get_data(picks=ecg_channel, return_times=True)
        data = data.flatten()
        
        # Simple visualization
        plt.figure(figsize=(15, 5))
        plt.plot(times[:10000], data[:10000])
        plt.title(f"ECG Signal from {file_path}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.savefig("ecg_sample.png")
        print(f"Saved sample visualization to ecg_sample.png")
        
        # Basic signal analysis
        sampling_rate = raw.info['sfreq']
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Signal duration: {len(data)/sampling_rate:.2f} seconds ({len(data)/sampling_rate/60:.2f} minutes)")
        print(f"Signal range: {np.min(data):.4f} to {np.max(data):.4f}")
        print(f"Signal mean: {np.mean(data):.4f}")
        print(f"Signal std: {np.std(data):.4f}")
        
        # For compatibility with existing code, create a record-like object
        class RecordLike:
            def __init__(self, data, sampling_rate, ch_names):
                self.p_signal = data.reshape(1, -1).T  # Match WFDB format
                self.fs = sampling_rate
                self.sig_name = ch_names
                self.n_sig = len(ch_names)
                self.sig_len = len(data)
                
        record = RecordLike(data, sampling_rate, [ecg_channel])
        
        # Continue with the analysis and generate a PDF report
        with PdfPages(output_pdf) as pdf:
            # Plot the raw signal
            fig, ax = plt.subplots(figsize=(12, 6))
            # Only plot the first 10 seconds for clarity
            plot_seconds = 10
            plot_samples = int(sampling_rate * plot_seconds)
            ax.plot(times[:plot_samples], data[:plot_samples])
            ax.set_title("Raw ECG Signal (First 10 seconds)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            pdf.savefig()
            plt.close()
            
            # Detect R-peaks and ECG features with neurokit2
            signals, info = nk.ecg_process(data, sampling_rate=sampling_rate)
            
            # Plot R-peaks on a segment of the signal
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(times[:plot_samples], data[:plot_samples], label="ECG Signal")
            
            # Find R-peaks that fall within our plot window
            rpeaks = info['ECG_R_Peaks']
            rpeaks_in_window = rpeaks[rpeaks < plot_samples]
            ax.scatter(times[rpeaks_in_window], data[rpeaks_in_window], color='red', label="R-peaks")
            
            ax.set_title("ECG Signal with R-peaks (First 10 seconds)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)
            pdf.savefig()
            plt.close()
            
            # Heart rate analysis
            if len(rpeaks) >= 2:
                # Calculate RR intervals
                rr_intervals = np.diff(rpeaks) / sampling_rate
                heart_rate = 60 / rr_intervals
                
                # Plot heart rate variability
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(heart_rate)
                ax.set_title("Heart Rate Over Time")
                ax.set_xlabel("Beat Number")
                ax.set_ylabel("Heart Rate (BPM)")
                ax.axhline(y=60, color='r', linestyle='--', label="60 BPM")
                ax.axhline(y=100, color='r', linestyle='--', label="100 BPM")
                ax.grid(True)
                pdf.savefig()
                plt.close()
                
                # Poincaré plot (RRn vs RRn+1)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(rr_intervals[:-1], rr_intervals[1:], alpha=0.5)
                ax.set_title("Poincaré Plot (RR intervals)")
                ax.set_xlabel("RR interval (s)")
                ax.set_ylabel("Next RR interval (s)")
                ax.grid(True)
                
                # Add identity line
                min_rr = min(rr_intervals)
                max_rr = max(rr_intervals)
                ax.plot([min_rr, max_rr], [min_rr, max_rr], 'r--')
                
                pdf.savefig()
                plt.close()
                
                # Calculate HRV metrics
                sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
                rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Root mean square of successive differences
                
                # Create a summary page with key metrics
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.axis('off')
                metrics_text = f"""
                ECG Analysis Summary:
                ------------------------------------------------
                Recording Duration: {len(data)/sampling_rate/60:.2f} minutes
                Sampling Rate: {sampling_rate} Hz
                Total Beats Detected: {len(rpeaks)}
                Average Heart Rate: {np.mean(heart_rate):.1f} BPM
                Minimum Heart Rate: {np.min(heart_rate):.1f} BPM
                Maximum Heart Rate: {np.max(heart_rate):.1f} BPM
                
                Heart Rate Variability Metrics:
                ------------------------------------------------
                SDNN: {sdnn:.4f} s
                RMSSD: {rmssd:.4f} s
                """
                ax.text(0.1, 0.5, metrics_text, va='center', fontsize=10, family='monospace')
                pdf.savefig()
                plt.close()
            
            print(f"Analysis complete. Report saved to {output_pdf}")
        
        # Return the data for further processing
        return {
            "signal": data,
            "sampling_rate": sampling_rate,
            "duration": len(data)/sampling_rate,
            "times": times
        }
        
    except Exception as e:
        print(f"Error analyzing EDF file: {str(e)}")
        raise RuntimeError(f"Failed to analyze EDF file: {e}")

def analyze_record(record):
    """Analyze an ECG record and detect arrhythmias."""
    # Extract ECG signal and sampling rate
    ecg_signal = record.p_signal[:, 0]  # Assume the first signal is ECG
    fs = record.fs  # Sampling frequency
    
    # Process the ECG signal using neurokit2
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    
    # Extract R-peaks and calculate heart rate
    rpeaks = info['ECG_R_Peaks']
    
    # Calculate RR intervals (in seconds)
    rr_intervals = np.diff(rpeaks) / fs
    
    # Calculate heart rate
    heart_rate = 60 / rr_intervals
    
    # HRV metrics for arrhythmia detection
    sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Root mean square of successive differences
    
    # Calculate pNN50
    nn50 = sum(abs(np.diff(rr_intervals)) > 0.05)  # Number of pairs of successive intervals differing by more than 50 ms
    pnn50 = nn50 / len(rr_intervals)
    
    # Identify potential arrhythmias
    arrhythmias = {}
    
    # Atrial Fibrillation criteria: Irregular rhythm and RMSSD above threshold
    if rmssd > 0.2 and sdnn > 0.1:
        arrhythmias['atrial_fibrillation'] = {
            'probability': min(1.0, (rmssd - 0.1) * 5),  # Simple scaling
            'evidence': f"High RMSSD ({rmssd:.3f}) and SDNN ({sdnn:.3f})"
        }
    
    # Bradycardia criteria: Heart rate < 60 BPM
    bradycardia_beats = sum(heart_rate < 60)
    if bradycardia_beats > len(heart_rate) * 0.1:  # If more than 10% of beats are bradycardic
        arrhythmias['bradycardia'] = {
            'probability': min(1.0, bradycardia_beats / len(heart_rate) * 2),
            'evidence': f"{bradycardia_beats} beats below 60 BPM ({bradycardia_beats/len(heart_rate)*100:.1f}%)"
        }
    
    # Tachycardia criteria: Heart rate > 100 BPM
    tachycardia_beats = sum(heart_rate > 100)
    if tachycardia_beats > len(heart_rate) * 0.1:  # If more than 10% of beats are tachycardic
        arrhythmias['tachycardia'] = {
            'probability': min(1.0, tachycardia_beats / len(heart_rate) * 2),
            'evidence': f"{tachycardia_beats} beats above 100 BPM ({tachycardia_beats/len(heart_rate)*100:.1f}%)"
        }
    
    # Create results dictionary
    results = {
        'heart_rate': {
            'mean': np.mean(heart_rate),
            'min': np.min(heart_rate),
            'max': np.max(heart_rate),
            'rr_intervals': rr_intervals.tolist()
        },
        'hrv': {
            'sdnn': sdnn,
            'rmssd': rmssd,
            'pnn50': pnn50
        },
        'arrhythmias': arrhythmias
    }
    
    return results

def plot_ecg_with_features(signal, fs, rpeaks=None, title="ECG Signal"):
    """Plot ECG signal with features."""
    # Create time array
    time = np.arange(len(signal)) / fs
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ECG signal
    ax.plot(time, signal, label="ECG")
    
    # Plot R-peaks if available
    if rpeaks is not None:
        ax.scatter(time[rpeaks], signal[rpeaks], color='red', label="R-peaks")
    
    # Add labels and grid
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    
    return fig

def detect_arrhythmias(ecg_signal, fs):
    """Detect arrhythmias in ECG signal."""
    # Process ECG signal
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    
    # Extract R-peaks
    rpeaks = info['ECG_R_Peaks']
    
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
        'types': []
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
    
    return arrhythmias, rpeaks, heart_rate

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_pdf = "ecg_analysis_report.pdf"
        if len(sys.argv) > 2:
            output_pdf = sys.argv[2]
            
        try:
            analyze_ecg_edf(file_path, output_pdf)
        except Exception as e:
            print(f"Error analyzing ECG: {str(e)}")
    else:
        print("Usage: python ecg_arrhythmia_analysis.py <edf_file_path> [output_pdf]")
        print("Example: python ecg_arrhythmia_analysis.py sample.edf report.pdf")