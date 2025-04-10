import streamlit as st
import subprocess
import os
import sys
import webbrowser
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ECG Analysis Suite Launcher",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.app-box {
    background-color: #1E1E1E;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #3F3F3F;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}
.app-title {
    color: #2196F3;
    font-size: 24px;
    margin-bottom: 10px;
    border-bottom: 1px solid #3F3F3F;
    padding-bottom: 8px;
}
.app-description {
    color: #FFFFFF;
    margin-bottom: 15px;
    font-size: 16px;
    line-height: 1.5;
}
.docs-link {
    font-size: 14px;
    color: #757575;
    text-decoration: none;
}
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 30px;
    color: #2196F3;
    padding-bottom: 10px;
    border-bottom: 2px solid #3F3F3F;
}
.tag {
    background-color: #2C2C2C;
    color: #42A5F5;
    padding: 5px 10px;
    border-radius: 20px;
    margin-right: 5px;
    font-size: 12px;
    font-weight: 500;
}
.button-row {
    display: flex;
    gap: 10px;
}

/* Button styling */
.stButton > button {
    background-color: #4CAF50 !important;
    color: white !important;
    border: none !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background-color: #45a049 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# App information
app_data = [
    {
        "title": "AFDx: Advanced ECG & AF Detection",
        "description": "A specialized tool for detecting and analyzing atrial fibrillation in ECG signals. Based on advanced algorithms from research by Camm et al. (2023), Kirchhof et al. (2022), and Hindricks et al. (2021) on HRV analysis and AF pattern recognition.",
        "file": "af_detection_app.py",
        "tags": ["Atrial Fibrillation", "Classification", "HRV Analysis"],
        "port": 8521,
        "docs": "docs/af_detection_app_overview.md"
    },
    {
        "title": "ECG Dashboard",
        "description": "A visualization-focused dashboard for ECG data with multiple views, heart rate trends, and rhythm analysis. Great for exploring patterns in ECG recordings.",
        "file": "ecg_dashboard.py",
        "tags": ["Visualization", "Trends", "Interactive"],
        "port": 8522,
        "docs": "docs/ecg_dashboard.md"
    },
    {
        "title": "ECG DeepDive: Signal to Insight",
        "description": "Advanced ECG analysis with over 100 cardiac biomarkers, detailed feature extraction, and multi-classifier approach for comprehensive signal analysis.",
        "file": "enhanced_ecg_app.py",
        "tags": ["Advanced Analysis", "Feature Extraction", "Multi-Classifier"],
        "port": 8523,
        "docs": "docs/enhanced_ecg_app.md"
    },
    {
        "title": "Holter ECG Review",
        "description": "A comprehensive web application for analyzing ECG data from EDF files with a focus on arrhythmia detection and medical reporting.",
        "file": "ecg_streamlit_app.py",
        "tags": ["EDF Analysis", "Medical Reporting", "Holter"],
        "port": 8524,
        "docs": "docs/ecg_streamlit_app.md"
    }
]

# Function to launch a Streamlit app
def launch_app(app_file, port):
    """Launch a Streamlit app in a new process and open browser."""
    # Check if app file exists
    if not os.path.exists(app_file):
        st.error(f"Application file {app_file} not found in the current directory.")
        return False
        
    try:
        # Get absolute path to the file
        app_file_abs = os.path.abspath(app_file)
        
        # Form the command to run the app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            app_file_abs, "--server.port", str(port), "--server.headless", "true"
        ]
        
        # Log the command for debugging
        print(f"Launching: {' '.join(cmd)}")
        
        # Kill any existing process on the same port
        try:
            import signal
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if any(f"--server.port {port}" in ' '.join(cmd_part) for cmd_part in [proc.info['cmdline']] if cmd_part):
                    print(f"Killing existing process on port {port}: {proc.info['pid']}")
                    os.kill(proc.info['pid'], signal.SIGTERM)
        except ImportError:
            print(f"Warning: Could not check for existing processes: psutil module not available")
        except (PermissionError, Exception) as e:
            print(f"Warning: Could not check for existing processes: {e}")
        
        # Run the process in the background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Wait a moment for the server to start
        st.info(f"Starting {os.path.basename(app_file)} on port {port}...")
        import time
        time.sleep(3)
        
        # Open the browser to the app
        app_url = f"http://localhost:{port}"
        webbrowser.open(app_url)
        
        return True
    except Exception as e:
        st.error(f"Error launching {app_file}: {str(e)}")
        return False

# Function to process documentation content
def process_documentation(content):
    """Replace any MIT License references with proprietary license information."""
    # Replace MIT License references
    content = content.replace("MIT License", "Proprietary License")
    content = content.replace("This application is distributed under the MIT License", 
                             "This application is proprietary software owned by IndaPoint Technologies Private Limited")
    content = content.replace("Open-source", "Proprietary")
    content = content.replace("open-source", "proprietary")
    content = content.replace("MIT", "Proprietary")
    
    # Add copyright footer if not present
    if "© 2025 IndaPoint Technologies" not in content:
        content += "\n\n---\n© 2025 IndaPoint Technologies Private Limited. All Rights Reserved.\nProprietary and Confidential."
    
    return content

# Main content
st.markdown("<h1 class='main-title'>ECG Analysis Suite</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("""
This launcher provides access to all ECG analysis applications in the suite. 
Each application serves a different purpose and offers unique features for ECG signal processing, 
visualization, and analysis. Select an application to launch it in a new browser tab.
""")

# App grid layout - 2 columns
col1, col2 = st.columns(2)

# Display apps in the grid
with col1:
    # First app
    st.markdown(f"<div class='app-box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-title'>{app_data[0]['title']}</div>", unsafe_allow_html=True)
    
    # Tags
    tags_html = ""
    for tag in app_data[0]['tags']:
        tags_html += f"<span class='tag'>{tag}</span>"
    st.markdown(tags_html, unsafe_allow_html=True)
    
    st.markdown(f"<div class='app-description'>{app_data[0]['description']}</div>", unsafe_allow_html=True)
    
    # Launch button
    if st.button(f"Launch {app_data[0]['title']}", key=app_data[0]['file']):
        with st.spinner(f"Launching {app_data[0]['title']}..."):
            success = launch_app(app_data[0]['file'], app_data[0]['port'])
            if success:
                st.success(f"{app_data[0]['title']} launched! Check your browser for a new tab.")
    
    # Documentation link
    docs_path = app_data[0]['docs']
    if os.path.exists(docs_path):
        with open(docs_path, 'r') as f:
            docs_content = f.read()
            # Process documentation to replace license references
            docs_content = process_documentation(docs_content)
        if st.button("View Documentation", key=f"docs_{app_data[0]['file']}"):
            st.markdown("### Documentation")
            st.markdown(docs_content)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Third app
    st.markdown(f"<div class='app-box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-title'>{app_data[2]['title']}</div>", unsafe_allow_html=True)
    
    # Tags
    tags_html = ""
    for tag in app_data[2]['tags']:
        tags_html += f"<span class='tag'>{tag}</span>"
    st.markdown(tags_html, unsafe_allow_html=True)
    
    st.markdown(f"<div class='app-description'>{app_data[2]['description']}</div>", unsafe_allow_html=True)
    
    # Launch button
    if st.button(f"Launch {app_data[2]['title']}", key=app_data[2]['file']):
        with st.spinner(f"Launching {app_data[2]['title']}..."):
            success = launch_app(app_data[2]['file'], app_data[2]['port'])
            if success:
                st.success(f"{app_data[2]['title']} launched! Check your browser for a new tab.")
    
    # Documentation link
    docs_path = app_data[2]['docs']
    if os.path.exists(docs_path):
        with open(docs_path, 'r') as f:
            docs_content = f.read()
            # Process documentation to replace license references
            docs_content = process_documentation(docs_content)
        if st.button("View Documentation", key=f"docs_{app_data[2]['file']}"):
            st.markdown("### Documentation")
            st.markdown(docs_content)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Second app
    st.markdown(f"<div class='app-box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-title'>{app_data[1]['title']}</div>", unsafe_allow_html=True)
    
    # Tags
    tags_html = ""
    for tag in app_data[1]['tags']:
        tags_html += f"<span class='tag'>{tag}</span>"
    st.markdown(tags_html, unsafe_allow_html=True)
    
    st.markdown(f"<div class='app-description'>{app_data[1]['description']}</div>", unsafe_allow_html=True)
    
    # Launch button
    if st.button(f"Launch {app_data[1]['title']}", key=app_data[1]['file']):
        with st.spinner(f"Launching {app_data[1]['title']}..."):
            success = launch_app(app_data[1]['file'], app_data[1]['port'])
            if success:
                st.success(f"{app_data[1]['title']} launched! Check your browser for a new tab.")
    
    # Documentation link
    docs_path = app_data[1]['docs']
    if os.path.exists(docs_path):
        with open(docs_path, 'r') as f:
            docs_content = f.read()
            # Process documentation to replace license references
            docs_content = process_documentation(docs_content)
        if st.button("View Documentation", key=f"docs_{app_data[1]['file']}"):
            st.markdown("### Documentation")
            st.markdown(docs_content)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Fourth app
    st.markdown(f"<div class='app-box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-title'>{app_data[3]['title']}</div>", unsafe_allow_html=True)
    
    # Tags
    tags_html = ""
    for tag in app_data[3]['tags']:
        tags_html += f"<span class='tag'>{tag}</span>"
    st.markdown(tags_html, unsafe_allow_html=True)
    
    st.markdown(f"<div class='app-description'>{app_data[3]['description']}</div>", unsafe_allow_html=True)
    
    # Launch button
    if st.button(f"Launch {app_data[3]['title']}", key=app_data[3]['file']):
        with st.spinner(f"Launching {app_data[3]['title']}..."):
            success = launch_app(app_data[3]['file'], app_data[3]['port'])
            if success:
                st.success(f"{app_data[3]['title']} launched! Check your browser for a new tab.")
    
    # Documentation link
    docs_path = app_data[3]['docs']
    if os.path.exists(docs_path):
        with open(docs_path, 'r') as f:
            docs_content = f.read()
            # Process documentation to replace license references
            docs_content = process_documentation(docs_content)
        if st.button("View Documentation", key=f"docs_{app_data[3]['file']}"):
            st.markdown("### Documentation")
            st.markdown(docs_content)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar information
st.sidebar.title("About ECG Analysis Suite")
st.sidebar.info("""
This suite of applications provides comprehensive tools for ECG signal processing, 
visualization, and analysis, with a special focus on atrial fibrillation detection.

Each application serves different purposes but shares common underlying libraries and tools.
""")

st.sidebar.title("Requirements")
st.sidebar.markdown("""
- Python 3.7+
- Streamlit
- NumPy/Pandas
- Matplotlib/Plotly
- NeuroKit2 (optional)
- BiospPy (optional)
""")

st.sidebar.title("Documentation")
st.sidebar.markdown("""
Comprehensive documentation for each application is available in the docs folder.
You can view documentation directly from each app card by clicking "View Documentation".
""")

# License information
st.sidebar.title("License")
st.sidebar.info("""
© 2025 IndaPoint Technologies Private Limited. All Rights Reserved.
Proprietary and Confidential. Unauthorized copying of this software, via any medium is strictly prohibited.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #757575; font-size: 12px;">
    ECG Analysis Suite Launcher • © 2025 IndaPoint Technologies Private Limited • All Rights Reserved • Proprietary Software
</div>
""", unsafe_allow_html=True) 