import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from io import BytesIO

# Page config with jazzy theme
st.set_page_config(
    page_title="🎯 VisionPro AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for jazzy UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

def detect_objects_opencv(image):
    """Basic object detection using OpenCV classical methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use Haar cascades for face detection (built into OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Edge detection for general objects
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_image = image.copy()
    detections = {}
    
    # Draw face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(annotated_image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw eye rectangles
    for (x, y, w, h) in eyes:
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated_image, 'Eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Count large contours as objects
    large_objects = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small contours
            large_objects += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # Create detection summary
    if len(faces) > 0:
        detections['Faces'] = len(faces)
    if len(eyes) > 0:
        detections['Eyes'] = len(eyes)
    if large_objects > 0:
        detections['Objects'] = large_objects
    
    return annotated_image, detections

def count_people_basic(image):
    """Count people using basic face detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    annotated_image = image.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated_image, f'Person {len(faces)}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return len(faces), annotated_image

def detect_text_regions(image):
    """Detect text regions using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply morphological operations to find text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = 0
    annotated_image = image.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small areas
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter based on text-like properties
            if 0.2 < aspect_ratio < 10:
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(annotated_image, 'Text', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                text_regions += 1
    
    return text_regions, annotated_image

def analyze_image_properties(image):
    """Analyze basic image properties"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate basic statistics
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Color analysis
    colors = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    dominant_hue = np.median(colors[:, :, 0])
    
    return {
        'brightness': mean_brightness,
        'contrast': contrast,
        'edge_density': edge_density * 100,
        'dominant_hue': dominant_hue
    }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 VisionPro AI</h1>
        <p>Computer Vision for Daily Tasks - Streamlit Cloud Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 Features")
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode",
        ["🔍 Object Detection", "📊 Inventory Counter", "👥 People Counter", 
         "📝 Text Detection", "🎨 Image Analysis", "📈 Analytics Dashboard"]
    )
    
    if app_mode == "🔍 Object Detection":
        st.markdown('<div class="feature-card"><h2>🔍 Object Detection</h2><p>Detect faces, eyes, and objects using OpenCV</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detected Objects")
                with st.spinner("Analyzing image..."):
                    annotated_image, detections = detect_objects_opencv(image_np)
                    
                st.image(annotated_image, use_column_width=True)
                
                if detections:
                    st.subheader("Detection Results")
                    for obj, count in detections.items():
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{obj}</h3>
                            <h2>{count}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Save to history
                    detection_entry = {
                        'timestamp': datetime.now(),
                        'detections': detections,
                        'total_objects': sum(detections.values())
                    }
                    st.session_state.detection_history.append(detection_entry)
                else:
                    st.info("No objects detected with current methods. Try an image with faces or clear objects.")
    
    elif app_mode == "📊 Inventory Counter":
        st.markdown('<div class="feature-card"><h2>📊 Inventory Counter</h2><p>Count items using contour detection</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload inventory image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Inventory Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Item Count")
                with st.spinner("Counting items..."):
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours by size
                    items = []
                    annotated_image = image_np.copy()
                    
                    for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        if 500 < area < 50000:  # Filter reasonable item sizes
                            items.append(area)
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(annotated_image, f'Item {len(items)}', (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                st.image(annotated_image, use_column_width=True)
                
                # Create inventory summary
                if items:
                    inventory_df = pd.DataFrame({
                        'Item': [f'Item {i+1}' for i in range(len(items))],
                        'Size': items,
                        'Category': ['Detected Object'] * len(items)
                    })
                    
                    st.dataframe(inventory_df, use_container_width=True)
                    
                    # Size distribution chart
                    fig = px.histogram(inventory_df, x='Size', title="Item Size Distribution")
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Total Items Detected", len(items))
                else:
                    st.info("No items detected. Try adjusting the image or using items with clear boundaries.")
    
    elif app_mode == "👥 People Counter":
        st.markdown('<div class="feature-card"><h2>👥 People Counter</h2><p>Count people using face detection</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload image with people", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("People Detection")
                with st.spinner("Counting people..."):
                    people_count, annotated_image = count_people_basic(image_np)
                
                st.image(annotated_image, use_column_width=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>People Detected</h3>
                    <h2>{people_count}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                if people_count == 0:
                    st.info("No faces detected. This method works best with clear, front-facing photos.")
    
    elif app_mode == "📝 Text Detection":
        st.markdown('<div class="feature-card"><h2>📝 Text Detection</h2><p>Detect text regions using morphological operations</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload image with text", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Text Regions")
                with st.spinner("Detecting text..."):
                    text_count, annotated_image = detect_text_regions(image_np)
                
                st.image(annotated_image, use_column_width=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Text Regions</h3>
                    <h2>{text_count}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    elif app_mode == "🎨 Image Analysis":
        st.markdown('<div class="feature-card"><h2>🎨 Image Analysis</h2><p>Analyze image properties and characteristics</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload image for analysis", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
                
                # Basic image info
                st.write(f"**Dimensions:** {image_np.shape[1]} x {image_np.shape[0]}")
                st.write(f"**Channels:** {image_np.shape[2] if len(image_np.shape) > 2 else 1}")
            
            with col2:
                st.subheader("Analysis Results")
                with st.spinner("Analyzing image properties..."):
                    props = analyze_image_properties(image_np)
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.metric("Brightness", f"{props['brightness']:.1f}")
                    st.metric("Contrast", f"{props['contrast']:.1f}")
                
                with col2b:
                    st.metric("Edge Density", f"{props['edge_density']:.1f}%")
                    st.metric("Dominant Hue", f"{props['dominant_hue']:.0f}°")
                
                # Color histogram
                fig = go.Figure()
                for i, color in enumerate(['red', 'green', 'blue']):
                    hist = np.histogram(image_np[:,:,i], bins=50, range=(0, 255))[0]
                    fig.add_trace(go.Scatter(y=hist, name=color.upper(), 
                                           line=dict(color=color)))
                
                fig.update_layout(title="Color Distribution", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "📈 Analytics Dashboard":
        st.markdown('<div class="feature-card"><h2>📈 Analytics Dashboard</h2><p>View detection history and analytics</p></div>', unsafe_allow_html=True)
        
        if st.session_state.detection_history:
            # Create analytics data
            all_detections = []
            timestamps = []
            total_counts = []
            
            for entry in st.session_state.detection_history:
                timestamps.append(entry['timestamp'])
                total_counts.append(entry['total_objects'])
                for obj, count in entry['detections'].items():
                    all_detections.extend([obj] * count)
            
            # Detection frequency chart
            if all_detections:
                detection_counts = Counter(all_detections)
                df_freq = pd.DataFrame([
                    {'Object': obj, 'Frequency': count}
                    for obj, count in detection_counts.items()
                ])
                
                fig_bar = px.bar(df_freq, x='Object', y='Frequency', 
                               title="Object Detection Frequency")
                fig_bar.update_layout(template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Timeline chart
            if len(timestamps) > 1:
                df_timeline = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Total Objects': total_counts
                })
                
                fig_line = px.line(df_timeline, x='Timestamp', y='Total Objects',
                                 title="Detection Timeline")
                fig_line.update_layout(template="plotly_white")
                st.plotly_chart(fig_line, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", len(st.session_state.detection_history))
            with col2:
                st.metric("Total Objects Detected", sum(total_counts))
            with col3:
                st.metric("Unique Object Types", len(set(all_detections)))
        else:
            st.info("No detection history available. Try using other features first!")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.detection_history = []
            st.success("History cleared!")
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎯 VisionPro AI - Streamlit Cloud Edition | Built with OpenCV & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
