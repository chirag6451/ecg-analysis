import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import tempfile
import os
from datetime import datetime, timedelta
import hashlib

# Import ECG arrhythmia classifier
try:
    from ecg_arrhythmia_classification import ECGArrhythmiaClassifier
    from ecg_holter_analysis import HolterAnalyzer
    modules_available = True
except ImportError:
    modules_available = False

# Page configuration
st.set_page_config(
    page_title="Atrial Fibrillation Detector",
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
.af-high {
    background-color: #ffcdd2;
    color: #b71c1c;
    padding: 0.5rem;
    border-radius: 4px;
    font-weight: bold;
}
.af-medium {
    background-color: #ffecb3;
    color: #ff6f00;
    padding: 0.5rem;
    border-radius: 4px;
    font-weight: bold;
}
.af-low {
    background-color: #c8e6c9;
    color: #1b5e20;
    padding: 0.5rem;
    border-radius: 4px;
    font-weight: bold;
}
.metric-card {
    background-color: #f5f5f5;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}
.ecg-plot {
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}
</style>
""", unsafe_allow_html=True)

def plot_ecg(signal, sampling_rate=200, title="ECG Signal"):
    """Plot ECG signal using matplotlib."""
    # Calculate time axis
    time = np.arange(len(signal)) / sampling_rate
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, signal, linewidth=1.5, color='#d32f2f')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig

def plot_af_gauge(af_prob):
    """Create a gauge chart for AF probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=af_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Atrial Fibrillation Probability (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#c8e6c9'},  # Green for low
                {'range': [30, 70], 'color': '#ffecb3'},  # Yellow for medium
                {'range': [70, 100], 'color': '#ffcdd2'}  # Red for high
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': af_prob * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def plot_rr_intervals(r_peaks, sampling_rate=200):
    """Plot RR intervals as a tachogram."""
    if len(r_peaks) < 2:
        return None
    
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) / sampling_rate
    beats = np.arange(len(rr_intervals))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(beats, rr_intervals, 'o-', markersize=6, color='#1976d2')
    ax.set_title("RR Interval Tachogram", fontsize=16)
    ax.set_xlabel("Beat Number", fontsize=14)
    ax.set_ylabel("RR Interval (seconds)", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at mean RR interval
    mean_rr = np.mean(rr_intervals)
    ax.axhline(mean_rr, color='red', linestyle='--', alpha=0.7, 
               label=f"Mean: {mean_rr:.3f}s ({60/mean_rr:.1f} BPM)")
    
    # Add horizontal lines at ±SDNN
    sdnn = np.std(rr_intervals)
    ax.axhline(mean_rr + sdnn, color='green', linestyle=':', alpha=0.7, 
               label=f"SDNN: {sdnn:.3f}s")
    ax.axhline(mean_rr - sdnn, color='green', linestyle=':', alpha=0.7)
    
    ax.legend(loc='best')
    fig.tight_layout()
    
    return fig

def plot_lorenz(rr_intervals):
    """Plot Lorenz/Poincaré plot of RR intervals."""
    if len(rr_intervals) < 3:
        return None
    
    # Create RR(n) vs RR(n+1)
    rr_n = rr_intervals[:-1]
    rr_n_plus_1 = rr_intervals[1:]
    
    # Calculate identity line range
    min_rr = min(np.min(rr_n), np.min(rr_n_plus_1))
    max_rr = max(np.max(rr_n), np.max(rr_n_plus_1))
    identity_line = np.array([min_rr, max_rr])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(rr_n, rr_n_plus_1, marker='o', color='#d32f2f', alpha=0.6, 
               edgecolors='black', s=50)
    ax.plot(identity_line, identity_line, 'b--', alpha=0.5, label="Identity Line")
    
    # Add ellipse fitting - if scipy is available
    try:
        from matplotlib.patches import Ellipse
        from scipy.stats import chi2
        
        # Calculate covariance and mean
        cov = np.cov(rr_n, rr_n_plus_1)
        mean_rr_n = np.mean(rr_n)
        mean_rr_n_plus_1 = np.mean(rr_n_plus_1)
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Calculate angle of ellipse
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        # Create ellipse for 95% confidence
        chi2_val = chi2.ppf(0.95, 2)
        width, height = 2 * np.sqrt(chi2_val * eigenvals)
        
        # Add ellipse to plot
        ellipse = Ellipse(xy=(mean_rr_n, mean_rr_n_plus_1),
                          width=width, height=height,
                          angle=angle, edgecolor='blue', facecolor='none',
                          linestyle='-', linewidth=2)
        ax.add_patch(ellipse)
        
        # Calculate SD1 and SD2 (standard deviations along minor and major axis)
        sd1 = np.sqrt(eigenvals[0])
        sd2 = np.sqrt(eigenvals[1])
        
        # Add to plot title
        plot_title = f"Poincaré Plot - SD1: {sd1:.3f}s, SD2: {sd2:.3f}s"
    except ImportError:
        plot_title = "Poincaré Plot"
    
    ax.set_title(plot_title, fontsize=16)
    ax.set_xlabel("RR(n) (seconds)", fontsize=14)
    ax.set_ylabel("RR(n+1) (seconds)", fontsize=14)
    ax.set_xlim(min_rr - 0.05, max_rr + 0.05)
    ax.set_ylim(min_rr - 0.05, max_rr + 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Make aspect ratio equal
    ax.set_aspect('equal')
    
    fig.tight_layout()
    return fig

def analyze_ecg_signal(signal, sampling_rate=200):
    """Analyze ECG signal for atrial fibrillation."""
    # Initialize classifier
    classifier = ECGArrhythmiaClassifier()
    
    # Run AF detection
    af_prob, af_metrics = classifier.detect_af(signal, sampling_rate=sampling_rate)
    
    # Get detailed R-peak information
    try:
        # Ensure signal is preprocessed for better peak detection
        processed_signal = classifier.preprocess_signal_for_rpeaks(signal, sampling_rate)
        
        # Detect R-peaks using the classifier's methods
        from biosppy.signals import ecg as bsp_ecg
        _, r_peaks = bsp_ecg.hamilton_segmenter(processed_signal, sampling_rate)
        r_peaks = bsp_ecg.correct_rpeaks(processed_signal, r_peaks, sampling_rate)
    except:
        # Fallback to simpler peak detection
        try:
            from scipy.signal import find_peaks
            r_peaks, _ = find_peaks(signal, distance=sampling_rate*0.5)
        except:
            r_peaks = []
    
    # Calculate RR intervals if enough peaks
    rr_intervals = np.diff(r_peaks) / sampling_rate if len(r_peaks) > 1 else []
    
    return af_prob, af_metrics, r_peaks, rr_intervals

def display_af_result(af_prob, af_metrics):
    """Display AF detection results."""
    # Display AF probability with gauge
    st.plotly_chart(plot_af_gauge(af_prob), use_container_width=True)
    
    # Display classification result
    if af_prob >= 0.7:
        st.markdown(f"<div class='af-high'>High probability of Atrial Fibrillation: {af_prob:.1%}</div>", 
                   unsafe_allow_html=True)
        st.markdown("""
        ### Clinical Interpretation
        This ECG segment shows characteristics highly suggestive of Atrial Fibrillation:
        - Irregular RR intervals with high variability
        - Absence of consistent P waves
        - Chaotic rhythm pattern
        
        Recommend clinical correlation and further evaluation by a healthcare professional.
        """)
    elif af_prob >= 0.3:
        st.markdown(f"<div class='af-medium'>Moderate probability of Atrial Fibrillation: {af_prob:.1%}</div>", 
                   unsafe_allow_html=True)
        st.markdown("""
        ### Clinical Interpretation
        This ECG segment shows some characteristics that may be consistent with Atrial Fibrillation:
        - Moderately irregular RR intervals
        - Some abnormal variability in heart rhythm
        - Possible atrial conduction abnormalities
        
        Consider further monitoring or evaluation by a healthcare professional.
        """)
    else:
        st.markdown(f"<div class='af-low'>Low probability of Atrial Fibrillation: {af_prob:.1%}</div>", 
                   unsafe_allow_html=True)
        st.markdown("""
        ### Clinical Interpretation
        This ECG segment shows a relatively regular rhythm without significant 
        characteristics of Atrial Fibrillation.
        """)
    
    # Display metrics in a formatted way
    st.markdown("### Heart Rate Variability Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Mean Heart Rate", f"{af_metrics.get('mean_hr', 0):.1f} BPM")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Rhythm Irregularity", f"{af_metrics.get('irregularity', 0):.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("RMSSD", f"{af_metrics.get('rmssd', 0):.3f} s")
        st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("SDNN", f"{af_metrics.get('sdnn', 0):.3f} s")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("pNN50", f"{af_metrics.get('pnn50', 0)*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

def display_hrv_plots(r_peaks, rr_intervals, sampling_rate=200):
    """Display HRV analysis plots."""
    if len(r_peaks) < 3:
        st.warning("Not enough R-peaks detected for detailed HRV analysis.")
        return
    
    st.markdown("### Heart Rate Variability Analysis")
    
    # RR interval tachogram
    st.markdown("#### RR Interval Tachogram")
    st.markdown("The tachogram shows how the time between heartbeats (RR intervals) varies over time. Regular patterns suggest normal rhythm, while irregularity may indicate arrhythmias like AF.")
    tachogram_fig = plot_rr_intervals(r_peaks, sampling_rate)
    st.pyplot(tachogram_fig)
    
    # Poincaré plot
    st.markdown("#### Poincaré Plot")
    st.markdown("The Poincaré plot shows each RR interval against the next one. A tight, cigar-shaped cluster along the identity line indicates normal rhythm. A scattered, circular pattern suggests AF.")
    poincare_fig = plot_lorenz(rr_intervals)
    if poincare_fig:
        st.pyplot(poincare_fig)
    else:
        st.warning("Could not generate Poincaré plot - insufficient RR intervals.")

def load_edf_segment(file, start_minute=0, duration_seconds=60):
    """Load a segment from an EDF file."""
    # Use HolterAnalyzer to handle EDF files
    analyzer = HolterAnalyzer()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    # Load the EDF file
    if analyzer.load_edf_file(tmp_path):
        # Get segment
        df = analyzer.get_segment(start_minute, duration_seconds)
        
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return df, analyzer.fs
    else:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
        return None, None

def load_csv_data(file, sampling_rate=200):
    """Load ECG data from a CSV file."""
    try:
        df = pd.read_csv(file)
        
        # Check if we have expected columns
        if 'time' in df.columns and 'signal' in df.columns:
            # Already has the expected format
            pass
        elif len(df.columns) >= 2:
            # Assume first column is time, second is signal
            df.columns = ['time', 'signal'] + list(df.columns[2:])
        elif len(df.columns) == 1:
            # Only one column - assume it's the signal
            signal = df.iloc[:, 0].values
            df = pd.DataFrame({
                'time': np.arange(len(signal)) / sampling_rate,
                'signal': signal
            })
        
        return df, sampling_rate
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

def display_detailed_af_analysis(signal, sampling_rate=200):
    """Display detailed information about AF classification and features."""
    st.markdown("## Detailed Atrial Fibrillation Analysis")
    
    # Create classifier
    classifier = ECGArrhythmiaClassifier()
    
    # Get basic AF detection
    af_prob, af_metrics = classifier.detect_af(signal, sampling_rate=sampling_rate)
    
    # Get more detailed classification
    predictions = classifier.predict(signal, sampling_rate=sampling_rate)
    probabilities = classifier.predict_proba(signal, sampling_rate=sampling_rate)
    
    # Display AF metrics in an expander for cleaner UI
    with st.expander("AF Detection Metrics", expanded=True):
        # Create two columns for metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### HRV Metrics")
            metrics_md = "<div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'>"
            for metric, value in af_metrics.items():
                if isinstance(value, (int, float)):
                    # Format the metric names for better readability
                    metric_name = metric.replace('_', ' ').title()
                    if metric == 'mean_hr':
                        metrics_md += f"<p><b>{metric_name}:</b> {value:.1f} BPM</p>"
                    elif metric == 'pnn50':
                        metrics_md += f"<p><b>{metric_name}:</b> {value*100:.1f}%</p>" 
                    else:
                        metrics_md += f"<p><b>{metric_name}:</b> {value:.4f}</p>"
            metrics_md += "</div>"
            st.markdown(metrics_md, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Atrial Fibrillation Probability Analysis")
            # Create probability bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            bar_color = 'green' if af_prob < 0.3 else 'orange' if af_prob < 0.7 else 'red'
            ax.bar(['AF Probability'], [af_prob], color=bar_color, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk Threshold')
            ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
            ax.legend()
            st.pyplot(fig)
    
    # Display detailed arrhythmia classification
    with st.expander("Arrhythmia Classification Results", expanded=True):
        # Count occurrences of each class
        if len(predictions) > 0:
            unique_classes, counts = np.unique(predictions, return_counts=True)
            class_percentages = counts / len(predictions) * 100
            
            # Create a dataframe with class information
            classification_data = []
            for cls, percentage in zip(unique_classes, class_percentages):
                class_name = classifier.get_class_name(cls)
                classification_data.append({
                    'Class ID': int(cls),
                    'Class Name': class_name,
                    'Percentage': f"{percentage:.1f}%",
                    'Count': int(counts[np.where(unique_classes == cls)[0][0]])
                })
            
            # Display classification table
            st.markdown("### ECG Classification Distribution")
            st.dataframe(pd.DataFrame(classification_data))
            
            # Create pie chart of classifications
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(
                counts, 
                labels=[classifier.get_class_name(cls) for cls in unique_classes],
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                colors=plt.cm.tab10.colors[:len(unique_classes)]
            )
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.setp(autotexts, size=10, weight="bold")
            st.pyplot(fig)
            
            # Add interpretation
            st.markdown("### Interpretation")
            af_class_percentage = 0
            for data in classification_data:
                if 'Atrial Fibrillation' in data['Class Name']:
                    af_class_percentage = float(data['Percentage'].strip('%'))
            
            if af_class_percentage > 50:
                st.error("This ECG segment shows predominant Atrial Fibrillation pattern.")
            elif af_class_percentage > 20:
                st.warning("This ECG segment shows significant Atrial Fibrillation activity.")
            elif af_class_percentage > 0:
                st.info("This ECG segment shows some Atrial Fibrillation activity.")
            else:
                st.success("This ECG segment does not show significant Atrial Fibrillation patterns.")
        else:
            st.warning("Could not perform detailed classification on this ECG segment.")
    
    # Display ECG feature information
    with st.expander("ECG Feature Analysis", expanded=True):
        features = classifier.preprocess_ecg(signal, sampling_rate=sampling_rate)
        
        if features.shape[0] > 0:
            # Calculate average feature values
            avg_features = np.mean(features, axis=0)
            
            # Feature names
            feature_names = [
                "Mean Amplitude", "Standard Deviation", "Maximum", "Minimum", "25th Percentile", 
                "75th Percentile", "Median", "Total Power", "Low Frequency Power", 
                "Mid Frequency Power", "High Frequency Power", "SDNN", "ASDNN", 
                "RR Ratio", "Kurtosis"
            ]
            
            # Display features
            st.markdown("### ECG Features Used for Classification")
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': avg_features
            })
            st.dataframe(feature_df)
            
            # Create basic feature bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            # Select a subset of interesting features for visualization
            selected_indices = [0, 1, 7, 11, 12, 13]  # Interesting features for AF
            selected_features = [feature_names[i] for i in selected_indices]
            selected_values = [avg_features[i] for i in selected_indices]
            
            ax.bar(selected_features, selected_values, color='royalblue')
            ax.set_ylabel('Value')
            ax.set_title('Key ECG Features')
            plt.xticks(rotation=45, ha='right')
            fig.tight_layout()
            st.pyplot(fig)
            
            # Add explanation about what these features mean for AF detection
            st.markdown("""
            ### Feature Interpretation for AF Detection
            
            **Key Features for Atrial Fibrillation:**
            
            1. **SDNN (Standard Deviation of NN intervals)** - Higher values indicate greater heart rate variability, 
               which is often observed in atrial fibrillation. AF typically shows SDNN values >0.1s.
            
            2. **ASDNN (Average of Standard Deviations)** - Similar to SDNN, elevated values suggest irregular rhythm.
            
            3. **RR Ratio (Max/Min RR interval)** - In AF, the ratio between the longest and shortest RR intervals 
               is typically high due to the irregularity of the rhythm. Values >1.5 often suggest AF.
            
            4. **Total Power** - Represents the overall variability of the heart rate. AF often shows higher values.
            
            5. **Standard Deviation** - Higher standard deviation in the ECG signal can indicate the irregularity 
               typical of AF patterns.
               
            6. **Kurtosis** - Measures the "tailedness" of the distribution of RR intervals. AF often shows 
               higher kurtosis values than normal sinus rhythm.
            """)
        else:
            st.warning("Could not extract features from this ECG segment.")
    
    # Educational section about AF
    with st.expander("Educational Information About Atrial Fibrillation", expanded=True):
        st.markdown("""
        ## Understanding Atrial Fibrillation

        ### What is Atrial Fibrillation?
        Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia, characterized by rapid, 
        irregular, and chaotic electrical activity in the atria, leading to uncoordinated atrial contractions 
        and irregular ventricular response.

        ### Clinical Significance
        - AF affects approximately 2-4% of adults over 65 years
        - Increases risk of stroke by 5-fold
        - Associated with increased risk of heart failure and mortality
        - Quality of life is often significantly impaired

        ### ECG Characteristics of AF
        - **Absence of P waves**: Regular P waves are replaced by rapid oscillations or fibrillatory waves
        - **Irregular R-R intervals**: The time between QRS complexes varies unpredictably
        - **Narrow QRS complexes**: Unless there is concurrent bundle branch block or aberrant conduction

        ### Types of Atrial Fibrillation
        1. **Paroxysmal AF**: Episodes that terminate spontaneously within 7 days
        2. **Persistent AF**: Episodes lasting longer than 7 days or requiring intervention
        3. **Long-standing persistent AF**: Continuous AF lasting over 12 months
        4. **Permanent AF**: When rhythm control strategies are abandoned

        ### Risk Factors
        - Advanced age
        - Hypertension
        - Heart failure
        - Coronary artery disease
        - Valvular heart disease
        - Diabetes mellitus
        - Obesity
        - Sleep apnea
        - Hyperthyroidism

        ### Management Approaches
        - Rate control medications
        - Rhythm control strategies
        - Anticoagulation to prevent stroke
        - Catheter ablation
        - Lifestyle modifications
        """)
        
        # Add an educational image about AF
        st.markdown("""
        ### Normal Sinus Rhythm vs. Atrial Fibrillation
        
        In normal sinus rhythm, there is a clear P wave before each QRS complex, and the RR intervals 
        are regular. In atrial fibrillation, P waves are absent and replaced by fibrillatory waves, 
        and the RR intervals are irregularly irregular.
        """)
        
        # Create a simple diagram to illustrate AF vs Normal
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Create time axis
        t = np.arange(0, 5, 0.01)
        
        # Normal sinus rhythm simulation
        normal_signal = np.zeros_like(t)
        p_wave_locations = np.arange(0.4, 5, 0.8)
        qrs_locations = np.arange(0.6, 5, 0.8)
        
        for loc in p_wave_locations:
            # Add P wave (Gaussian curve)
            normal_signal += 0.5 * np.exp(-((t - loc) ** 2) / 0.003)
        
        for loc in qrs_locations:
            # Add QRS complex (sharper Gaussian)
            normal_signal += 1.5 * np.exp(-((t - loc) ** 2) / 0.001) - 0.5 * np.exp(-((t - loc - 0.02) ** 2) / 0.002)
        
        # AF simulation (irregular rhythm)
        af_signal = np.zeros_like(t)
        # Random fibrillatory waves (small, rapid oscillations)
        fibrillatory_waves = 0.2 * np.sin(2 * np.pi * 8 * t) + 0.1 * np.sin(2 * np.pi * 12 * t)
        
        # Irregular QRS complexes
        irregular_qrs_locations = [0.5, 0.9, 1.7, 2.1, 2.8, 3.4, 3.7, 4.6]
        
        for loc in irregular_qrs_locations:
            # Add QRS complex (sharper Gaussian)
            af_signal += 1.5 * np.exp(-((t - loc) ** 2) / 0.001) - 0.5 * np.exp(-((t - loc - 0.02) ** 2) / 0.002)
        
        # Add fibrillatory activity
        af_signal += fibrillatory_waves
        
        # Plot
        ax1.plot(t, normal_signal, 'b-')
        ax1.set_title('Normal Sinus Rhythm')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim(0, 4)
        ax1.set_xticks([])
        ax1.text(0.05, 1.5, 'Regular P waves', color='red', fontsize=10)
        ax1.text(0.05, -0.5, 'Regular RR intervals', color='green', fontsize=10)
        
        ax2.plot(t, af_signal, 'r-')
        ax2.set_title('Atrial Fibrillation')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.set_xlim(0, 4)
        ax2.set_xticks([])
        ax2.text(0.05, 1.5, 'Absence of P waves, fibrillatory waves', color='red', fontsize=10)
        ax2.text(0.05, -0.5, 'Irregular RR intervals', color='green', fontsize=10)
        
        fig.tight_layout()
        st.pyplot(fig)

def main():
    st.title("Atrial Fibrillation Detection Tool")
    st.markdown("""
    This application analyzes ECG data to detect atrial fibrillation, a common heart rhythm disorder. 
    Upload your ECG data in EDF or CSV format to get a comprehensive analysis.
    """)
    
    # Check if required modules are available
    if not modules_available:
        st.error("""
        Required modules are not available. Please make sure the following are installed:
        - ecg_arrhythmia_classification.py
        - ecg_holter_analysis.py
        
        These modules should be in the same directory as this app or in your Python path.
        """)
        return
    
    # Sidebar for data upload and configuration
    st.sidebar.header("Data Upload")
    
    # File upload options
    file_type = st.sidebar.radio("Select input file type", ["EDF (Holter)", "CSV/TXT"])
    
    # File uploader
    if file_type == "EDF (Holter)":
        uploaded_file = st.sidebar.file_uploader("Upload EDF file", type=["edf"])
        
        if uploaded_file is not None:
            # EDF configuration
            st.sidebar.header("EDF Configuration")
            
            # Show loading status
            status_placeholder = st.sidebar.empty()
            status_placeholder.info(f"Loading EDF file ({uploaded_file.size/1048576:.1f} MB). Please wait...")
            
            # Load segment from EDF
            df, fs = load_edf_segment(uploaded_file)
            
            if df is not None:
                status_placeholder.success("EDF file loaded successfully")
                
                # Segment selection
                st.sidebar.subheader("Segment Selection")
                
                # Calculate max minutes, ensuring at least 1 minute difference between min and max
                # This fixes the RangeError when min and max are equal
                max_minutes = max(1, int(len(df) / fs / 60) - 1)  # Ensure at least 1
                
                start_minute = st.sidebar.slider("Start time (minutes)", 0, max_minutes, 0)
                duration_seconds = st.sidebar.slider("Duration (seconds)", 10, 300, 60)
                
                # Re-load with selected segment
                df, fs = load_edf_segment(uploaded_file, start_minute, duration_seconds)
                
                if df is not None:
                    signal = df['signal'].values
                    
                    # Display ECG signal
                    st.markdown("### ECG Signal")
                    st.markdown("<div class='ecg-plot'>", unsafe_allow_html=True)
                    ecg_fig = plot_ecg(signal, fs, f"ECG Signal at {start_minute} minutes")
                    st.pyplot(ecg_fig)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Analysis options
                    analysis_type = st.radio(
                        "Select analysis mode",
                        ["Basic Analysis", "Detailed Analysis", "Both"],
                        index=2  # Default to Both
                    )
                    
                    # Run selected analysis
                    if st.button("Analyze for Atrial Fibrillation", key="analyze_edf"):
                        with st.spinner("Analyzing ECG for atrial fibrillation..."):
                            # Basic analysis
                            if analysis_type in ["Basic Analysis", "Both"]:
                                # Analyze the signal
                                af_prob, af_metrics, r_peaks, rr_intervals = analyze_ecg_signal(signal, fs)
                                
                                # Display AF results
                                display_af_result(af_prob, af_metrics)
                                
                                # Display HRV plots
                                display_hrv_plots(r_peaks, rr_intervals, fs)
                            
                            # Detailed analysis
                            if analysis_type in ["Detailed Analysis", "Both"]:
                                display_detailed_af_analysis(signal, fs)
                else:
                    st.error("Could not extract segment from EDF file")
            else:
                status_placeholder.error("Failed to load EDF file. Make sure it's a valid EDF format.")
    
    else:  # CSV/TXT
        uploaded_file = st.sidebar.file_uploader("Upload CSV/TXT file", type=["csv", "txt"])
        
        if uploaded_file is not None:
            # CSV configuration
            st.sidebar.header("CSV Configuration")
            
            # Sampling frequency
            fs = st.sidebar.number_input("Sampling Rate (Hz)", 100, 1000, 200)
            
            # Load CSV data
            df, fs = load_csv_data(uploaded_file, fs)
            
            if df is not None:
                st.sidebar.success(f"CSV file loaded with {len(df)} data points")
                
                signal = df['signal'].values
                
                # Display ECG signal
                st.markdown("### ECG Signal")
                st.markdown("<div class='ecg-plot'>", unsafe_allow_html=True)
                ecg_fig = plot_ecg(signal, fs, "ECG Signal")
                st.pyplot(ecg_fig)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Analysis options
                analysis_type = st.radio(
                    "Select analysis mode",
                    ["Basic Analysis", "Detailed Analysis", "Both"],
                    index=2  # Default to Both
                )
                
                # Run selected analysis
                if st.button("Analyze for Atrial Fibrillation", key="analyze_csv"):
                    with st.spinner("Analyzing ECG for atrial fibrillation..."):
                        # Basic analysis
                        if analysis_type in ["Basic Analysis", "Both"]:
                            # Analyze the signal
                            af_prob, af_metrics, r_peaks, rr_intervals = analyze_ecg_signal(signal, fs)
                            
                            # Display AF results
                            display_af_result(af_prob, af_metrics)
                            
                            # Display HRV plots
                            display_hrv_plots(r_peaks, rr_intervals, fs)
                        
                        # Detailed analysis
                        if analysis_type in ["Detailed Analysis", "Both"]:
                            display_detailed_af_analysis(signal, fs)
    
    # Information and instructions in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Information")
    st.sidebar.markdown("""
    ### What is Atrial Fibrillation?
    
    Atrial fibrillation (AF) is an irregular heart rhythm characterized by:
    - Rapid, irregular heartbeats
    - Disorganized electrical signals in the atria
    - Absence of consistent P waves on ECG
    
    ### Interpreting Results
    
    - **High probability (>70%)**: Strong evidence of AF
    - **Moderate probability (30-70%)**: Some evidence of AF
    - **Low probability (<30%)**: Limited evidence of AF
    
    *Note: This tool provides decision support but does not replace clinical diagnosis.*
    """)

if __name__ == "__main__":
    main() 