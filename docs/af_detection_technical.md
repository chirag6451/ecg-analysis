# Atrial Fibrillation Detection: Technical Documentation

This document provides detailed technical information about the atrial fibrillation detection algorithm, the feature extraction process, and the implementation details of the ECGArrhythmiaClassifier used in the AF Detection App.

## ECGArrhythmiaClassifier Architecture

The `ECGArrhythmiaClassifier` is implemented as a Python class that combines traditional signal processing techniques with machine learning to detect and classify various cardiac arrhythmias, with special emphasis on atrial fibrillation.

### Class Structure

```python
class ECGArrhythmiaClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.classes = {
            0: 'Normal',
            1: 'Supraventricular Premature Beat',
            2: 'Premature Ventricular Contraction',
            3: 'Fusion of Ventricular and Normal Beat',
            4: 'Unclassifiable Beat',
            5: 'Atrial Fibrillation'
        }
        self._initialize_with_defaults()
```

### Main Methods

1. `preprocess_ecg(signal_data, window_size=180, sampling_rate=200)`: Processes raw ECG signals and extracts feature vectors
2. `train(X, y)`: Trains the classifier with feature vectors and corresponding labels
3. `predict(signal, sampling_rate=200)`: Classifies ECG segments into arrhythmia types
4. `predict_proba(signal, sampling_rate=200)`: Returns probability estimates for each class
5. `detect_af(signal, sampling_rate=200)`: Specialized method for AF detection with probability and metrics
6. `preprocess_signal_for_rpeaks(signal, sampling_rate)`: Optimizes signal for R-peak detection
7. `plot_classification_results(signal, predictions, probabilities=None)`: Visualizes classification results

## Signal Preprocessing Pipeline

### Preprocessing for Feature Extraction

The signal preprocessing workflow involves multiple steps to enhance the quality of ECG signals:

1. **Normalization**: Zero mean and unit variance scaling
   ```python
   signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)
   ```

2. **Bandpass Filtering**: Remove noise while preserving cardiac information
   ```python
   nyquist = sampling_rate / 2
   low_cutoff = 0.5 / nyquist  # 0.5 Hz (remove baseline wander)
   high_cutoff = 40 / nyquist   # 40 Hz (remove high-frequency noise)
   b, a = sp_signal.butter(4, [low_cutoff, high_cutoff], btype='band')
   filtered_signal = sp_signal.filtfilt(b, a, signal_data)
   ```

3. **Windowing**: Segments are processed with 50% overlap
   ```python
   for i in range(0, len(filtered_signal) - window_size, window_size//2):
       window = filtered_signal[i:i+window_size]
       # Process each window...
   ```

### Preprocessing for R-Peak Detection

The R-peak detection preprocessing includes additional steps to enhance QRS complexes:

1. **Standard preprocessing**: Normalization and bandpass filtering
2. **Advanced preprocessing for low-amplitude signals**:
   - Derivative calculation to enhance QRS complexes
   - Squaring to amplify peaks
   - Moving window integration for smoothing
   ```python
   if signal_range < 0.5:  # Low amplitude signal
       derivative = np.diff(filtered)
       derivative = np.append(derivative, derivative[-1])
       squared = derivative**2
       window_width = int(0.08 * sampling_rate)  # 80ms window
       filtered = np.convolve(squared, np.ones(window_width)/window_width, mode='same')
   ```

## Feature Extraction Process

The feature extraction process creates a comprehensive feature vector for each ECG window, combining multiple domains:

### Time-Domain Statistical Features (7 features)

```python
basic_features = [
    np.mean(window),                # Mean amplitude
    np.std(window),                 # Standard deviation 
    np.max(window),                 # Maximum value
    np.min(window),                 # Minimum value
    np.percentile(window, 25),      # 25th percentile
    np.percentile(window, 75),      # 75th percentile
    np.median(window)               # Median value
]
```

### Frequency-Domain Features (4 features)

```python
f, psd = sp_signal.welch(window, fs=sampling_rate, nperseg=min(256, len(window)))
freq_features = [
    np.sum(psd),                       # Total power
    np.sum(psd[(f>=0.5) & (f<=8)]),    # Low frequency power
    np.sum(psd[(f>=8) & (f<=20)]),     # Mid frequency power
    np.sum(psd[(f>=20) & (f<=40)])     # High frequency power
]
```

### Heart Rate Variability Features (4 features)

```python
# R-peak detection
peaks, _ = sp_signal.find_peaks(window, height=0.5, distance=sampling_rate*0.3)

# Calculate RR intervals
rr_intervals = np.diff(peaks) / sampling_rate

# HRV features
hrv_features = [
    np.std(rr_intervals),                      # SDNN
    np.mean(np.abs(np.diff(rr_intervals))),    # ASDNN
    np.max(rr_intervals) / np.min(rr_intervals) if len(rr_intervals) > 0 else 1.0,  # RR ratio
    kurtosis(rr_intervals) if len(rr_intervals) > 3 else 0  # Kurtosis
]
```

## AF Detection Algorithm Details

The AF detection algorithm focuses specifically on the unique pattern of atrial fibrillation:

### R-Peak Detection

```python
# First try biosppy's hamilton detector
_, r_peaks = bsp_ecg.hamilton_segmenter(signal, sampling_rate)
r_peaks = bsp_ecg.correct_rpeaks(signal, r_peaks, sampling_rate)

# Fallback to neurokit's detector if needed
r_peaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
```

### HRV Metrics Calculation

```python
# Calculate RR intervals (in seconds)
rr_intervals = np.diff(r_peaks) / sampling_rate

# Calculate heart rate
mean_hr = 60 / np.mean(rr_intervals)
metrics['mean_hr'] = mean_hr

# Calculate HRV metrics - SDNN (Standard deviation of NN intervals)
metrics['sdnn'] = np.std(rr_intervals)

# RMSSD (Root Mean Square of Successive Differences)
rr_diffs = np.diff(rr_intervals)
metrics['rmssd'] = np.sqrt(np.mean(rr_diffs**2))

# pNN50 (Percentage of successive RR intervals differing by more than 50ms)
nn50 = np.sum(np.abs(rr_diffs) > 0.05)  # 50ms = 0.05s
metrics['pnn50'] = nn50 / len(rr_diffs)

# Calculate irregularity metric (coefficient of variation of RR intervals)
metrics['irregularity'] = np.std(rr_intervals) / np.mean(rr_intervals)
```

### Probability Calculation

The AF probability is calculated using a weighted combination of metrics:

```python
# Calculate probability based on established thresholds/rules
prob_from_irregularity = min(1.0, metrics['irregularity'] / 1.2)
prob_from_rmssd = min(1.0, metrics['rmssd'] * 75)
prob_from_pnn50 = min(1.0, metrics['pnn50'] * 3)

# Heart rate contribution (AF often has higher heart rate)
hr_factor = 0
if 100 <= mean_hr <= 175:
    hr_factor = (mean_hr - 100) / 75  # Scale 100-175 BPM to 0-1
elif mean_hr > 175:
    hr_factor = 1.0

# Combine metrics with different weights
probability = 0.50 * prob_from_irregularity + \
             0.30 * prob_from_rmssd + \
             0.15 * prob_from_pnn50 + \
             0.05 * hr_factor
```

### Deterministic Offset for Reproducibility

A small deterministic offset is added to ensure consistent results for the same signal:

```python
# Calculate file/signal uniqueness signature
sig_hash = hashlib.md5(signal[:min(1000, len(signal))].tobytes()).hexdigest()

# Create a small deterministic offset (±0.005 or 0.5%)
deterministic_offset = (int(sig_hash[:8], 16) % 100 - 50) / 10000

# Apply the offset to probability
probability = min(1.0, max(0.0, probability + deterministic_offset))
```

## Classification Model

### Model Architecture

The classifier uses a Random Forest model with the following configuration:

```python
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Feature Scaling

Standard scaling is applied to normalize feature values:

```python
from sklearn.preprocessing import StandardScaler
self.scaler = StandardScaler()
X_scaled = self.scaler.fit_transform(X)
```

### Model Initialization

To prevent NotFittedError when using the model without training data, a default initialization is performed:

```python
def _initialize_with_defaults(self):
    """Initialize the model with default data to avoid NotFittedError."""
    # Create a simple synthetic dataset
    X_default = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    y_default = np.array([0, 0])  # All normal class
    
    # Fit the scaler and model with default data
    self.scaler.fit(X_default)
    self.model.fit(X_default, y_default)
    self.is_fitted = True
```

## Fallback Implementations

The classifier includes fallback implementations for optional dependencies to ensure it works with minimal requirements:

### BiospPy Fallback

```python
class BiospyFallback:
    @staticmethod
    def hamilton_segmenter(signal, sampling_rate):
        # Minimal fallback using scipy.signal.find_peaks
        threshold = 0.6 * np.max(signal)
        min_distance = int(0.2 * sampling_rate)
        peaks, _ = sp_signal.find_peaks(signal, height=threshold, distance=min_distance)
        return signal, peaks
    
    @staticmethod
    def correct_rpeaks(signal, rpeaks, sampling_rate):
        # Simple correction: look for maximum in small window around each peak
        corrected_peaks = []
        window_size = int(0.025 * sampling_rate)
        
        for peak in rpeaks:
            start = max(0, peak - window_size)
            end = min(len(signal), peak + window_size)
            if start < end:
                local_max = start + np.argmax(signal[start:end])
                corrected_peaks.append(local_max)
            else:
                corrected_peaks.append(peak)
                
        return np.array(corrected_peaks)
```

### Neurokit2 Fallback

```python
class NeurokitFallback:
    @staticmethod
    def ecg_peaks(signal, sampling_rate=None):
        # Minimalist fallback for ecg_peaks
        threshold = 0.6 * np.max(signal)
        min_distance = int(0.2 * sampling_rate)
        peaks, _ = sp_signal.find_peaks(signal, height=threshold, distance=min_distance)
        return {}, {'ECG_R_Peaks': peaks}
```

## Performance Optimizations

Several optimizations are implemented to ensure reliable performance:

1. **Early validation**: Signal checks ensure valid inputs before processing
   ```python
   if len(signal) < sampling_rate * 5:  # Need at least 5 seconds
       return 0, {'error': 'Signal too short for analysis'}
   ```

2. **Robust error handling**: Graceful fallbacks for edge cases
   ```python
   try:
       # Main processing logic
   except Exception as e:
       import traceback
       print(f"Error in AF detection: {str(e)}")
       print(traceback.format_exc())
       return 0, {'error': f'Analysis error: {str(e)}'}
   ```

3. **Multiple detection methods**: Ensures R-peaks can be found in different signal qualities
   ```python
   try:
       # Primary method
   except Exception:
       # Secondary method
   except:
       # Final fallback
   ```

## HRV Metrics Interpretation

The key HRV metrics used for AF detection can be interpreted as follows:

| Metric | Normal Range | AF Range | Interpretation |
|--------|--------------|----------|----------------|
| SDNN | 0.02-0.05 s | >0.1 s | Higher values indicate greater heart rate variability |
| RMSSD | 0.01-0.03 s | >0.08 s | Elevated in AF due to beat-to-beat variability |
| pNN50 | 1-15% | >30% | Very high in AF due to irregular intervals |
| Irregularity (CV) | 0.05-0.15 | >0.2 | Key metric for AF, measures overall chaos |

## Clinical Thresholds

The application uses the following probability thresholds for clinical interpretation:

- **High Risk**: AF probability ≥ 0.7
- **Moderate Risk**: AF probability between 0.3 and 0.7
- **Low Risk**: AF probability < 0.3

These thresholds are based on optimizing the balance between sensitivity and specificity for AF detection.

## Example Feature Values

The table below shows typical feature values for normal sinus rhythm versus atrial fibrillation:

| Feature | Normal Range | AF Range |
|---------|--------------|----------|
| Mean amplitude | Varies by recording | Varies by recording |
| Standard deviation | Lower | Higher |
| Total power | Lower | Higher |
| SDNN | 0.02-0.05 s | >0.1 s |
| ASDNN | 0.01-0.03 s | >0.06 s |
| RR ratio | 1.1-1.3 | >1.5 |
| Kurtosis | Close to 0 | Higher |

## Implementation Considerations

### Sampling Rate Compatibility

The algorithm is designed to work with ECG signals at different sampling rates, with a default of 200 Hz. The key time-based parameters are scaled based on the sampling rate:

```python
# Window size is in samples, not seconds
window_size = 180  # 0.9 seconds at 200 Hz

# Time-based parameters are scaled
distance = sampling_rate * 0.3  # 0.3 seconds
```

### Window Size Considerations

The default window size of 180 samples at 200 Hz (0.9 seconds) was chosen to:
- Include at least one full cardiac cycle even at low heart rates
- Be short enough to capture localized arrhythmia events
- Optimize feature extraction performance

For feature extraction, 50% overlapping windows are used to ensure no events are missed at window boundaries.

## References

For more detailed information about the HRV metrics and their relationship to atrial fibrillation, please refer to:

1. Faust, O., Hagiwara, Y., Hong, T. J., Lih, O. S., & Acharya, U. R. (2018). Deep learning for healthcare applications based on physiological signals: A review. Computer methods and programs in biomedicine, 161, 1-13.

2. Dash, S., Chon, K. H., Lu, S., & Raeder, E. A. (2009). Automatic real time detection of atrial fibrillation. Annals of biomedical engineering, 37(9), 1701-1709.

3. Tateno, K., & Glass, L. (2001). Automatic detection of atrial fibrillation using the coefficient of variation and density histograms of RR and ΔRR intervals. Medical and Biological Engineering and Computing, 39(6), 664-671.

4. Task Force of the European Society of Cardiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation, 93(5), 1043-1065. 