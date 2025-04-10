import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt
import neurokit2 as nk

# Try to import biosppy, but provide fallback if tkinter is not available
try:
    import biosppy.signals.ecg as bsp_ecg
except ImportError as e:
    print(f"Warning: biosppy import failed: {e}")
    # Define a minimal fallback for bsp_ecg.hamilton_segmenter and bsp_ecg.correct_rpeaks
    class BiospyFallback:
        @staticmethod
        def hamilton_segmenter(signal, sampling_rate):
            """Minimal fallback implementation for R-peak detection."""
            # Use scipy.signal.find_peaks as a basic alternative
            # This is not as good as Hamilton's algorithm but will work as fallback
            threshold = 0.6 * np.max(signal)
            min_distance = int(0.2 * sampling_rate)  # 200ms minimum distance
            peaks, _ = signal.find_peaks(signal, height=threshold, distance=min_distance)
            return signal, peaks
        
        @staticmethod
        def correct_rpeaks(signal, rpeaks, sampling_rate):
            """Minimal fallback implementation for R-peak correction."""
            # Simple correction: look for maximum in small window around each peak
            corrected_peaks = []
            window_size = int(0.025 * sampling_rate)  # 25ms window
            
            for peak in rpeaks:
                start = max(0, peak - window_size)
                end = min(len(signal), peak + window_size)
                if start < end:
                    local_max = start + np.argmax(signal[start:end])
                    corrected_peaks.append(local_max)
                else:
                    corrected_peaks.append(peak)
                    
            return np.array(corrected_peaks)
    
    # Create fallback object
    bsp_ecg = BiospyFallback()

from scipy.fft import fft

class ECGFeatureExtractor:
    """Advanced feature extraction for ECG signals."""
    
    def __init__(self, fs=200):
        """
        Initialize feature extractor with sampling frequency.
        
        Args:
            fs (float): Sampling frequency in Hz
        """
        self.fs = fs
        
    def extract_all_features(self, ecg_signal):
        """
        Extract comprehensive feature set from ECG signal.
        
        Args:
            ecg_signal (np.array): Raw ECG signal
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        # Basic signal validation
        if len(ecg_signal) < 5 * self.fs:
            return {"error": "Signal too short for feature extraction"}
            
        # Normalize signal
        ecg_normalized = self._normalize_signal(ecg_signal)
        
        # Detect R-peaks using enhanced algorithm
        r_peaks = self._detect_r_peaks_enhanced(ecg_normalized)
        
        if len(r_peaks) < 3:
            return {"error": "Not enough R-peaks detected for feature extraction"}
            
        # Extract various feature sets
        features = {}
        
        # Time-domain HRV features
        features.update(self._extract_time_domain_hrv(r_peaks))
        
        # Frequency-domain HRV features
        features.update(self._extract_frequency_domain_hrv(r_peaks))
        
        # Non-linear HRV features
        features.update(self._extract_nonlinear_hrv(r_peaks))
        
        # Wavelet features
        features.update(self._extract_wavelet_features(ecg_normalized))
        
        # Morphology features
        features.update(self._extract_morphology_features(ecg_normalized, r_peaks))
        
        # Statistical features
        features.update(self._extract_statistical_features(ecg_normalized))
        
        return features
        
    def _normalize_signal(self, ecg_signal):
        """Normalize ECG signal to zero mean and unit standard deviation."""
        return (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-10)
        
    def _detect_r_peaks_enhanced(self, ecg_signal):
        """
        Enhanced R-peak detection using TERMA-inspired approach.
        
        This implements a fusion algorithm based on principles from the TERMA algorithm
        mentioned in the Nature paper.
        """
        try:
            # Apply bandpass filter to isolate QRS complex frequencies
            nyquist = self.fs / 2
            low = 5 / nyquist
            high = 15 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, ecg_signal)
            
            # Apply derivative to enhance QRS complex
            derivative = np.diff(filtered)
            derivative = np.append(derivative, derivative[-1])
            
            # Square the signal to make all values positive
            squared = derivative ** 2
            
            # Apply moving window integration
            window_size = int(0.08 * self.fs)  # 80ms window
            moving_avg = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
            
            # Dynamic thresholding: adapt threshold based on signal characteristics
            signal_range = np.max(moving_avg) - np.min(moving_avg)
            if signal_range < 0.01:  # Very low amplitude signal
                threshold = 0.05 * np.max(moving_avg)
            else:
                threshold = 0.3 * np.max(moving_avg)
                
            # Find peaks with adaptive distance based on estimated heart rate
            min_distance = int(0.2 * self.fs)  # Start with 200ms minimum distance
            peaks, _ = signal.find_peaks(moving_avg, height=threshold, distance=min_distance)
            
            # If few peaks found, try with lower threshold
            if len(peaks) < 5:
                threshold = 0.05 * np.max(moving_avg)
                peaks, _ = signal.find_peaks(moving_avg, height=threshold, distance=min_distance // 2)
            
            # If still too few peaks, return what we have
            if len(peaks) < 3:
                return peaks
                
            # Adjust for unrealistic heart rates
            rr_intervals = np.diff(peaks) / self.fs
            mean_hr = 60 / np.mean(rr_intervals)
            if mean_hr > 180:  # Unrealistically high HR
                # Take every other peak
                peaks = peaks[::2]
                
            # Refinement step: Use original signal to find exact R-peak locations
            # Look in a small window around each detected peak
            refined_peaks = []
            window_half_size = int(0.025 * self.fs)  # 25ms window half-size
            
            for peak in peaks:
                start = max(0, peak - window_half_size)
                end = min(len(ecg_signal), peak + window_half_size + 1)
                if start < end:
                    # Find the maximum value in the window on the original signal
                    window_max_idx = start + np.argmax(ecg_signal[start:end])
                    refined_peaks.append(window_max_idx)
            
            return np.array(refined_peaks)
            
        except Exception as e:
            print(f"Error in R-peak detection: {str(e)}")
            # Fallback to biosppy's r-peak detection
            try:
                _, r_peaks = bsp_ecg.hamilton_segmenter(ecg_signal, sampling_rate=self.fs)
                r_peaks = bsp_ecg.correct_rpeaks(ecg_signal, r_peaks, sampling_rate=self.fs)
                return r_peaks
            except:
                # Last resort: naive peak detection
                threshold = 0.7 * np.max(ecg_signal)
                min_distance = int(0.25 * self.fs)
                peaks, _ = signal.find_peaks(ecg_signal, height=threshold, distance=min_distance)
                return peaks
    
    def _extract_time_domain_hrv(self, r_peaks):
        """Extract time-domain HRV features from R-peaks."""
        if len(r_peaks) < 3:
            return {"time_domain_hrv_error": "Not enough R-peaks"}
            
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(r_peaks) / self.fs
        
        # HRV time-domain features
        features = {
            "mean_nn": np.mean(rr_intervals),
            "sdnn": np.std(rr_intervals),  # Standard deviation of RR intervals
            "rmssd": np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),  # Root mean square of successive differences
            "sdsd": np.std(np.diff(rr_intervals)),  # Standard deviation of successive differences
            "nn50": np.sum(np.abs(np.diff(rr_intervals)) > 0.05),  # Number of pairs of successive intervals differing by more than 50ms
            "pnn50": np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / max(1, len(rr_intervals)-1),  # Percentage of successive intervals differing by more than 50ms
            "nn20": np.sum(np.abs(np.diff(rr_intervals)) > 0.02),  # 20ms variant
            "pnn20": np.sum(np.abs(np.diff(rr_intervals)) > 0.02) / max(1, len(rr_intervals)-1),  # 20ms variant
            "hr_mean": 60 / np.mean(rr_intervals),  # Mean heart rate
            "hr_min": 60 / np.max(rr_intervals),    # Min heart rate
            "hr_max": 60 / np.min(rr_intervals),    # Max heart rate
        }
        
        return features
    
    def _extract_frequency_domain_hrv(self, r_peaks):
        """Extract frequency-domain HRV features from R-peaks."""
        if len(r_peaks) < 5:
            return {"freq_domain_hrv_error": "Not enough R-peaks"}
            
        try:
            # Calculate RR intervals in seconds
            rr_intervals = np.diff(r_peaks) / self.fs
            
            # Interpolate RR intervals to get evenly sampled signal
            # Create time array from cumulative sum of RR intervals
            time_rr = np.cumsum(rr_intervals)
            time_rr = np.insert(time_rr, 0, 0)  # Insert 0 at the beginning
            
            # Interpolate to 4Hz
            fs_interp = 4  # Hz
            time_interp = np.arange(0, time_rr[-1], 1/fs_interp)
            rr_interp = np.interp(time_interp, time_rr[:-1], rr_intervals)
            
            # Remove trend
            rr_interp = rr_interp - np.mean(rr_interp)
            
            # Apply Hanning window
            window = np.hanning(len(rr_interp))
            rr_interp = rr_interp * window
            
            # Calculate power spectral density
            fft_result = fft(rr_interp)
            fft_amplitude = np.abs(fft_result) ** 2
            freqs = np.fft.fftfreq(len(rr_interp), 1/fs_interp)
            
            # Calculate power in different frequency bands
            vlf_power = np.sum(fft_amplitude[(freqs >= 0.003) & (freqs < 0.04)])  # Very low frequency: 0.003-0.04 Hz
            lf_power = np.sum(fft_amplitude[(freqs >= 0.04) & (freqs < 0.15)])    # Low frequency: 0.04-0.15 Hz
            hf_power = np.sum(fft_amplitude[(freqs >= 0.15) & (freqs < 0.4)])     # High frequency: 0.15-0.4 Hz
            total_power = vlf_power + lf_power + hf_power
            
            # Return frequency-domain features
            features = {
                "vlf_power": vlf_power,
                "lf_power": lf_power,
                "hf_power": hf_power,
                "total_power": total_power,
                "lf_hf_ratio": lf_power / (hf_power + 1e-10),  # LF/HF ratio
                "lf_norm": 100 * lf_power / (lf_power + hf_power + 1e-10),  # Normalized LF
                "hf_norm": 100 * hf_power / (lf_power + hf_power + 1e-10),  # Normalized HF
            }
            
            return features
            
        except Exception as e:
            print(f"Error in frequency domain HRV: {str(e)}")
            return {"freq_domain_hrv_error": str(e)}
    
    def _extract_nonlinear_hrv(self, r_peaks):
        """Extract non-linear HRV features from R-peaks."""
        if len(r_peaks) < 5:
            return {"nonlinear_hrv_error": "Not enough R-peaks"}
            
        try:
            # Calculate RR intervals in seconds
            rr_intervals = np.diff(r_peaks) / self.fs
            
            # Poincaré plot features
            rr_n = rr_intervals[:-1]  # RR_n
            rr_n1 = rr_intervals[1:]  # RR_n+1
            
            # SD1: Standard deviation of points perpendicular to the line of identity
            sd1 = np.std(np.subtract(rr_n1, rr_n)) / np.sqrt(2)
            
            # SD2: Standard deviation of points along the line of identity
            sd2 = np.std(np.add(rr_n1, rr_n)) / np.sqrt(2)
            
            # Sample entropy (approximation)
            diff_rr = np.diff(rr_intervals)
            sample_entropy = -np.log(np.sum(np.abs(diff_rr) < 0.2 * np.std(rr_intervals)) / 
                                    (len(diff_rr) - 1))
            
            features = {
                "sd1": sd1,
                "sd2": sd2,
                "sd1_sd2_ratio": sd1 / (sd2 + 1e-10),
                "ellipse_area": np.pi * sd1 * sd2,
                "sample_entropy": sample_entropy
            }
            
            return features
            
        except Exception as e:
            print(f"Error in non-linear HRV: {str(e)}")
            return {"nonlinear_hrv_error": str(e)}
    
    def _extract_wavelet_features(self, ecg_signal):
        """Extract wavelet-based features from the ECG signal."""
        try:
            # Apply discrete wavelet transform
            coeffs = pywt.wavedec(ecg_signal, 'db4', level=4)
            
            # Extract features from each level
            features = {}
            for i, coeff in enumerate(coeffs):
                level_name = "approx" if i == 0 else f"detail_{i}"
                features[f"wavelet_{level_name}_mean"] = np.mean(coeff)
                features[f"wavelet_{level_name}_std"] = np.std(coeff)
                features[f"wavelet_{level_name}_energy"] = np.sum(coeff ** 2)
                features[f"wavelet_{level_name}_kurtosis"] = kurtosis(coeff)
                features[f"wavelet_{level_name}_skewness"] = skew(coeff)
            
            return features
            
        except Exception as e:
            print(f"Error in wavelet features: {str(e)}")
            return {"wavelet_error": str(e)}
    
    def _extract_morphology_features(self, ecg_signal, r_peaks):
        """Extract ECG morphology features related to P, QRS, T waves."""
        if len(r_peaks) < 3:
            return {"morphology_error": "Not enough R-peaks"}
            
        try:
            # Extract heartbeats around each R-peak
            beats = []
            for r_peak in r_peaks[1:-1]:  # Skip first and last to ensure we have enough signal
                # Define window size: 0.25s before and 0.45s after R-peak
                pre_window = int(0.25 * self.fs)
                post_window = int(0.45 * self.fs)
                
                if r_peak - pre_window >= 0 and r_peak + post_window < len(ecg_signal):
                    beat = ecg_signal[r_peak - pre_window:r_peak + post_window]
                    beats.append(beat)
            
            if not beats:
                return {"morphology_error": "Could not extract beats"}
                
            # Average beat
            avg_beat = np.mean(np.array(beats), axis=0)
            
            # Find Q, S, T points
            # Q-point: minimum before R-peak
            q_idx = np.argmin(avg_beat[:int(0.25 * self.fs)])
            
            # S-point: minimum after R-peak
            s_window = int(0.1 * self.fs)  # Look 100ms after R-peak
            s_idx = int(0.25 * self.fs) + np.argmin(avg_beat[int(0.25 * self.fs):int(0.25 * self.fs) + s_window])
            
            # T-peak: maximum after S-point (in the expected T-wave region)
            t_start = s_idx + int(0.08 * self.fs)  # Start looking 80ms after S
            t_end = min(len(avg_beat), s_idx + int(0.3 * self.fs))  # Look up to 300ms after S
            if t_start < t_end:
                t_idx = t_start + np.argmax(avg_beat[t_start:t_end])
            else:
                t_idx = s_idx + int(0.2 * self.fs)  # Default if window is invalid
            
            # Calculate intervals
            qrs_duration = (s_idx - q_idx) / self.fs * 1000  # in ms
            qt_interval = (t_idx - q_idx) / self.fs * 1000  # in ms
            st_segment = (t_idx - s_idx) / self.fs * 1000  # in ms
            
            # Calculate amplitudes
            r_amp = avg_beat[int(0.25 * self.fs)]  # R-peak amplitude
            q_amp = avg_beat[q_idx]  # Q amplitude
            s_amp = avg_beat[s_idx]  # S amplitude
            t_amp = avg_beat[t_idx]  # T amplitude
            
            # PR segment analysis (look for P-wave)
            pr_window = avg_beat[:q_idx]
            if len(pr_window) > int(0.05 * self.fs):
                p_idx = np.argmax(pr_window[:-int(0.05 * self.fs)])
                p_amp = pr_window[p_idx]
                pr_interval = (q_idx - p_idx) / self.fs * 1000  # in ms
            else:
                p_amp = 0
                pr_interval = 0
            
            features = {
                "qrs_duration": qrs_duration,
                "qt_interval": qt_interval,
                "st_segment": st_segment,
                "pr_interval": pr_interval,
                "r_amplitude": r_amp,
                "q_amplitude": q_amp,
                "s_amplitude": s_amp,
                "t_amplitude": t_amp,
                "p_amplitude": p_amp,
                "rs_ratio": r_amp / (abs(s_amp) + 1e-10),
                "rt_ratio": r_amp / (abs(t_amp) + 1e-10),
                "qrs_energy": np.sum(avg_beat[q_idx:s_idx] ** 2),
                "t_energy": np.sum(avg_beat[s_idx:t_idx] ** 2) if t_idx > s_idx else 0
            }
            
            return features
            
        except Exception as e:
            print(f"Error in morphology features: {str(e)}")
            return {"morphology_error": str(e)}
    
    def _extract_statistical_features(self, ecg_signal):
        """Extract statistical features from the ECG signal."""
        try:
            # Basic statistics
            features = {
                "mean": np.mean(ecg_signal),
                "std": np.std(ecg_signal),
                "var": np.var(ecg_signal),
                "kurtosis": kurtosis(ecg_signal),
                "skewness": skew(ecg_signal),
                "rms": np.sqrt(np.mean(ecg_signal ** 2)),
                "range": np.max(ecg_signal) - np.min(ecg_signal),
                "energy": np.sum(ecg_signal ** 2) / len(ecg_signal),
                "min": np.min(ecg_signal),
                "max": np.max(ecg_signal),
                "median": np.median(ecg_signal),
                "mode": float(np.bincount(np.round(ecg_signal * 100).astype(int)).argmax()) / 100
            }
            
            # Percentile features
            percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
            for p in percentiles:
                features[f"percentile_{p}"] = np.percentile(ecg_signal, p)
                
            return features
            
        except Exception as e:
            print(f"Error in statistical features: {str(e)}")
            return {"statistical_error": str(e)} 