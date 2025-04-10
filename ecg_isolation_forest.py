import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Optional imports if you have wfdb or kaggle
try:
    import wfdb
except ImportError:
    wfdb = None

try:
    from sklearn.datasets import fetch_openml
except ImportError:
    fetch_openml = None

# --------- CONFIG ----------
USE_SYNTHETIC = True
USE_ECG5000 = True
USE_KAGGLE = False  # Set to True if kaggle API is configured
USE_PHYSIONET = False  # Set to True if wfdb is installed
CONTAMINATION = 0.02

# --------- UTILITY: Plot Results ----------
def plot_results(df, signal_col='signal', title="ECG Anomaly Detection"):
    plt.figure(figsize=(14, 5))
    plt.plot(df['time'], df[signal_col], label='Signal')
    if 'is_anomaly' in df:
        plt.scatter(df[df['is_anomaly']]['time'], df[df['is_anomaly']][signal_col],
                    color='red', label='Anomalies')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------- SOURCE 1: Synthetic ECG ----------
def load_synthetic_ecg(length=1400):
    time = np.linspace(0, 10, length)
    signal = 0.6 * np.sin(1.75 * np.pi * time) + 0.3 * np.random.randn(length)
    anomalies = np.random.choice(length, size=20, replace=False)
    signal[anomalies] += np.random.uniform(3, 6, size=20)
    return pd.DataFrame({'time': time, 'signal': signal})

# --------- SOURCE 2: ECG5000 ----------
def load_ecg5000():
    if not fetch_openml:
        raise ImportError("scikit-learn is missing fetch_openml")
    ecg = fetch_openml(name='ECG5000', version=1, as_frame=True)
    data = ecg.frame
    # Remove label column and take first row (single time series)
    signal = data.iloc[0, 1:].values
    time = np.arange(len(signal))
    return pd.DataFrame({'time': time, 'signal': signal})

# --------- SOURCE 3: Kaggle Dataset (if API is set up) ----------
def load_kaggle_ecg(kaggle_dataset="shayanfazeli/heartbeat", use_fallback=True):
    """
    Load ECG data from Kaggle heartbeat dataset.
    Requires Kaggle API credentials to be set up or fallback option enabled.
    
    Args:
        kaggle_dataset (str): Kaggle dataset id
        use_fallback (bool): Whether to use fallback synthetic data if Kaggle fails
        
    Returns:
        pd.DataFrame: DataFrame with time and signal columns
    """
    # First check if CSV already exists (perhaps manually downloaded)
    csv_path = os.path.join("heartbeat", "mitbih_train.csv")
    if os.path.exists(csv_path):
        print(f"Using existing CSV file at {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        sample = df.iloc[0, :-1].values
        time = np.arange(len(sample))
        return pd.DataFrame({'time': time, 'signal': sample})
    
    # Try using Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Create directory if it doesn't exist
        if not os.path.exists("heartbeat"):
            os.makedirs("heartbeat", exist_ok=True)
            
            # Download dataset using Kaggle API
            print(f"Downloading dataset {kaggle_dataset}...")
            api.dataset_download_files(kaggle_dataset, path="heartbeat")
            
            # Extract the zip file
            zip_path = os.path.join("heartbeat", "heartbeat.zip")
            if os.path.exists(zip_path):
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("heartbeat")
            else:
                raise FileNotFoundError(f"Downloaded zip file not found at {zip_path}")
        
        # Check if the CSV file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        # Read the CSV file
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, header=None)
        
        # Get a single sample
        sample = df.iloc[0, :-1].values
        time = np.arange(len(sample))
        
        return pd.DataFrame({'time': time, 'signal': sample})
        
    except Exception as e:
        error_msg = f"Error loading Kaggle data: {str(e)}"
        print(error_msg)
        
        if use_fallback:
            print("Using synthetic data as fallback")
            return load_synthetic_ecg(length=187)  # Length matches ECG heartbeat size
        else:
            raise Exception(error_msg)

# --------- SOURCE 4: PhysioNet (MIT-BIH) ----------
def load_physionet_record(record_id='100', lead=0):
    """
    Load ECG data from PhysioNet MIT-BIH Arrhythmia Database.
    
    Args:
        record_id (str): Record ID (e.g., '100', '101', etc.)
        lead (int): ECG lead to use (0 for first lead)
        
    Returns:
        pd.DataFrame: DataFrame with time and signal columns
        dict: Metadata about the record
    """
    if not wfdb:
        raise ImportError("wfdb not installed. Use: pip install wfdb")
    
    try:
        # List available records in MIT-BIH database
        available_records = wfdb.get_record_list('mitdb')
        if record_id not in available_records:
            raise ValueError(f"Record {record_id} not found in MIT-BIH database. Available records: {available_records}")
        
        # Download and read the record
        record = wfdb.rdrecord(record_id, pn_dir='mitdb')
        
        # Get metadata
        metadata = {
            'record_name': record.record_name,
            'fs': record.fs,
            'n_sig': record.n_sig,
            'sig_len': record.sig_len,
            'sig_name': record.sig_name,
            'units': record.units,
            'comments': record.comments
        }
        
        # Extract signal and time
        if lead >= record.n_sig:
            raise ValueError(f"Lead {lead} not available. Available leads: {list(range(record.n_sig))}")
            
        signal = record.p_signal[:, lead]
        time = np.arange(len(signal)) / record.fs
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time,
            'signal': signal
        })
        
        return df, metadata
        
    except Exception as e:
        raise RuntimeError(f"Error loading PhysioNet record {record_id}: {str(e)}")

# --------- SOURCE 5: SHDB-AF Dataset (Atrial Fibrillation) ----------
def load_shdb_af_data(patient_id, segment_offset_minutes=0, segment_duration_seconds=60):
    """
    Load ECG data from the SHDB-AF (Saitama Heart Database - Atrial Fibrillation) dataset.
    
    Args:
        patient_id (int): Patient ID (1-143)
        segment_offset_minutes (int): Minutes from the start of the recording to begin the segment
        segment_duration_seconds (int): Duration of the segment to extract in seconds
        
    Returns:
        pd.DataFrame: DataFrame with time and signal columns
    """
    try:
        import wfdb
        
        # Format patient ID with leading zeros
        patient_str = f"{patient_id:03d}"
        
        # Path to the SHDB-AF record
        # Note: In your structure, files are directly in the shdb-af directory
        record_path = f"shdb-af/{patient_str}"
        
        # Check if the record exists
        if not os.path.exists(f"{record_path}.dat") or not os.path.exists(f"{record_path}.hea"):
            raise FileNotFoundError(f"SHDB-AF record {patient_str} not found. Expected path: {record_path}")
        
        # Calculate samples to skip and read
        sampling_rate = 200  # SHDB-AF sampling rate is 200 Hz
        start_sample = segment_offset_minutes * 60 * sampling_rate
        # The data is usually split in multiple segments, let's read more than requested to ensure we have enough
        read_samples = min(segment_duration_seconds * sampling_rate * 2, 24 * 60 * 60 * sampling_rate)
        
        # Read the record (only first channel - modified lead II)
        signals, fields = wfdb.rdsamp(record_path, channels=[0], sampfrom=start_sample, sampto=start_sample+read_samples)
        
        # Take only the requested duration
        signals = signals[:segment_duration_seconds * sampling_rate]
        
        # Create time array (in seconds)
        time = np.arange(len(signals)) / sampling_rate
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time,
            'signal': signals.flatten()
        })
        
        print(f"Loaded SHDB-AF patient {patient_str}: {len(df)} samples, duration: {len(df)/sampling_rate:.1f}s")
        
        return df
        
    except ImportError:
        raise ImportError("wfdb package not installed. Install with: pip install wfdb")
        
    except Exception as e:
        raise Exception(f"Error loading SHDB-AF data: {str(e)}")

# --------- Isolation Forest Detection ----------
def detect_anomalies(df, signal_col='signal', contamination=0.02):
    scaler = StandardScaler()
    df['scaled'] = scaler.fit_transform(df[[signal_col]])
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(df[['scaled']])
    df['is_anomaly'] = df['anomaly'] == -1
    return df, model

# --------- MAIN ----------
def main():
    data_sources = []

    if USE_SYNTHETIC:
        print("✅ Loading synthetic ECG data...")
        data_sources.append(("Synthetic", load_synthetic_ecg()))

    if USE_ECG5000:
        print("✅ Loading ECG5000 dataset from OpenML...")
        data_sources.append(("ECG5000", load_ecg5000()))

    if USE_KAGGLE:
        print("✅ Loading ECG from Kaggle...")
        data_sources.append(("Kaggle", load_kaggle_ecg()))

    if USE_PHYSIONET:
        print("✅ Loading record from PhysioNet MIT-BIH...")
        try:
            df, metadata = load_physionet_record(record_id='100', lead=0)
            print(f"Record metadata: {metadata}")
            data_sources.append(("PhysioNet", df))
        except Exception as e:
            print(f"❌ Error loading PhysioNet data: {e}")

    for name, df in data_sources:
        print(f"\n🔍 Running Isolation Forest on: {name}")
        df, model = detect_anomalies(df, contamination=CONTAMINATION)
        plot_results(df, title=f"{name} - ECG Anomaly Detection")

if __name__ == "__main__":
    main()
