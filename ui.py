import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import base64
from io import BytesIO

# Page configuration with custom theme
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, professional design with scan animation
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --background-dark: #0f172a;
        --background-light: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
    }
    
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #3b82f6 !important;
        border-radius: 10px;
        background: rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Image container with scan animation */
    .image-container {
        position: relative;
        display: inline-block;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin: 0 auto;
    }
    
    .image-wrapper {
        position: relative;
        display: inline-block;
        width: 100%;
    }
    
    .scan-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 10;
    }
    
    .scan-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(59, 130, 246, 0.3) 20%,
            rgba(59, 130, 246, 1) 50%, 
            rgba(59, 130, 246, 0.3) 80%,
            transparent 100%);
        box-shadow: 
            0 0 15px rgba(59, 130, 246, 0.8),
            0 0 30px rgba(59, 130, 246, 0.6),
            0 0 45px rgba(59, 130, 246, 0.4);
        opacity: 0;
    }
    
    .scan-line.active {
        animation: scan 2.5s ease-in-out;
    }
    
    .scan-grid {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        opacity: 0;
    }
    
    .scan-grid.active {
        animation: fadeInOut 2.5s ease-in-out;
    }
    
    @keyframes scan {
        0% {
            top: 0;
            opacity: 0;
        }
        5% {
            opacity: 1;
        }
        95% {
            opacity: 1;
        }
        100% {
            top: 100%;
            opacity: 0;
        }
    }
    
    @keyframes fadeInOut {
        0%, 100% {
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
    }
    
    /* Image styling */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-positive {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        border-left: 5px solid #991b1b;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-left: 5px solid #065f46;
    }
    
    .result-card h2 {
        color: white;
        font-size: 1.8rem;
        margin: 0 0 1rem 0;
        font-weight: 700;
    }
    
    .result-card p {
        color: #f1f5f9;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .confidence-bar {
        width: 100%;
        height: 30px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ffffff 0%, #e0e7ff 100%);
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #1e293b;
        transition: width 1s ease;
    }
    
    /* Info sections */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #cbd5e1;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>Brain Tumor Detection System</h1>
        <p>AI-Powered Medical Image Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/brain_tumor_model.keras")

model = load_model()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload MRI Scan")
    st.markdown('<div class="info-box">Please upload a brain MRI image in JPG, PNG, or JPEG format for analysis.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    
    # Clear results when new file is uploaded
    if uploaded_file and st.session_state.get('last_file_id') != id(uploaded_file):
        st.session_state.has_results = False
        st.session_state.is_analyzing = False
        st.session_state.last_file_id = id(uploaded_file)

with col2:
    if uploaded_file is not None:
        st.markdown("### Preview")
        image = Image.open(uploaded_file)
        
        # Convert image to base64 for HTML embedding
        img_base64 = image_to_base64(image)
        
        # Display image with scan animation overlay using HTML
        col2_1, col2_2, col2_3 = st.columns([1, 3, 1])
        with col2_2:
            # Add 'active' class to trigger animation when analyzing
            animation_class = "active" if st.session_state.is_analyzing else ""
            st.markdown(f"""
                <div class="image-container">
                    <img src="{img_base64}" style="width: 100%; border-radius: 15px; display: block;">
                    <div class="scan-overlay">
                        <div class="scan-grid {animation_class}"></div>
                        <div class="scan-line {animation_class}"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Analysis section
if uploaded_file is not None:
    st.markdown("---")
    
    # Center the button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("Analyze MRI Scan", use_container_width=True)
    
    if analyze_button:
        # Start scan animation
        st.session_state.is_analyzing = True
        st.rerun()
    
    # Run prediction if analyzing
    if st.session_state.is_analyzing:
        with st.spinner("Analyzing MRI scan..."):
            # Allow time for scan animation to complete
            time.sleep(2.5)
            
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image, verbose=0)[0][0]
            
            confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
            
            # Store results in session state
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.has_results = True
            
            # Stop scan animation
            st.session_state.is_analyzing = False
    
    # Display results if available
    if st.session_state.get('has_results', False):
        prediction = st.session_state.prediction
        confidence = st.session_state.confidence
        
        # Display results
        st.markdown("### Analysis Results")
        
        if prediction > 0.5:
            st.markdown(f"""
                <div class="result-card result-positive">
                    <h2>Tumor Detected</h2>
                    <p>The AI model has detected abnormalities consistent with a brain tumor.</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%;">
                            {confidence:.1f}% Confidence
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="info-box"><strong>Important:</strong> This is a preliminary analysis. Please consult with a qualified medical professional for proper diagnosis and treatment.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card result-negative">
                    <h2>No Tumor Detected</h2>
                    <p>The AI model did not detect significant abnormalities in the scan.</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%;">
                            {confidence:.1f}% Confidence
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="info-box"><strong>Note:</strong> While no tumor was detected, this result should be reviewed by a medical professional as part of comprehensive care.</div>', unsafe_allow_html=True)
        
        # Additional information
        with st.expander("View Technical Details"):
            st.write(f"**Prediction Score:** {prediction:.4f}")
            st.write(f"**Threshold:** 0.5")
            st.write(f"**Model Confidence:** {confidence:.2f}%")
            st.write(f"**Image Dimensions:** 224x224 (processed)")
else:
    # Welcome message when no file is uploaded
    st.markdown("---")
    st.markdown("""
        <div class="info-box">
            <h3>How to Use</h3>
            <ol>
                <li>Upload a brain MRI image using the file uploader above</li>
                <li>Preview the image to ensure it's loaded correctly</li>
                <li>Click the "Analyze MRI Scan" button to start the AI analysis</li>
                <li>Review the results and confidence score</li>
            </ol>
            <p><strong>Disclaimer:</strong> This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
        </div>
    """, unsafe_allow_html=True)
