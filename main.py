import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import os

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
    
    .sidebar .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

@st.cache_resource
def load_model():
    """Load YOLO model"""
    try:
        model = YOLO('yolov8n.pt')  # Using nano model for faster inference
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def detect_objects(image, model, confidence_threshold=0.5):
    """Perform object detection on image"""
    if model is None:
        return None, {}
    
    results = model(image, conf=confidence_threshold)
    
    # Extract detection info
    detections = {}
    annotated_image = image.copy()
    
    if len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                # Get class name
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Count detections
                if class_name in detections:
                    detections[class_name] += 1
                else:
                    detections[class_name] = 1
                
                # Draw bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return annotated_image, detections

def count_people(image, model):
    """Count people in image"""
    if model is None:
        return 0, image
    
    results = model(image, classes=[0])  # Class 0 is 'person' in COCO dataset
    annotated_image = image.copy()
    people_count = 0
    
    if len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            people_count = len(boxes)
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                
                # Draw bounding box for people
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"Person: {confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return people_count, annotated_image

def detect_text_regions(image):
    """Detect text regions using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use EAST text detector approach (simplified)
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
                text_regions += 1
    
    return text_regions, annotated_image

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 VisionPro AI</h1>
        <p>Advanced Computer Vision for Daily Tasks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 Features")
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode",
        ["🔍 Object Detection", "📊 Inventory Counter", "👥 People Counter", 
         "📝 Text Detection", "📈 Analytics Dashboard"]
    )
    
    # Load model
    model = load_model()
    
    if app_mode == "🔍 Object Detection":
        st.markdown('<div class="feature-card"><h2>🔍 Object Detection</h2><p>Detect and identify objects in images using state-of-the-art AI</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        
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
                    annotated_image, detections = detect_objects(image_np, model, confidence)
                    
                if annotated_image is not None:
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
    
    elif app_mode == "📊 Inventory Counter":
        st.markdown('<div class="feature-card"><h2>📊 Inventory Counter</h2><p>Count and track inventory items automatically</p></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload inventory image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Inventory Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Inventory Count")
                with st.spinner("Counting items..."):
                    annotated_image, detections = detect_objects(image_np, model, 0.5)
                
                if detections:
                    # Create inventory summary
                    inventory_df = pd.DataFrame([
                        {'Item': item, 'Count': count, 'Category': 'General'}
                        for item, count in detections.items()
                    ])
                    
                    st.dataframe(inventory_df, use_container_width=True)
                    
                    # Pie chart
                    fig = px.pie(inventory_df, values='Count', names='Item', 
                                title="Inventory Distribution")
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Total count
                    total_items = sum(detections.values())
                    st.metric("Total Items", total_items)
    
    elif app_mode == "👥 People Counter":
        st.markdown('<div class="feature-card"><h2>👥 People Counter</h2><p>Count people in images for crowd monitoring</p></div>', unsafe_allow_html=True)
        
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
                    people_count, annotated_image = count_people(image_np, model)
                
                st.image(annotated_image, use_column_width=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>People Detected</h3>
                    <h2>{people_count}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    elif app_mode == "📝 Text Detection":
        st.markdown('<div class="feature-card"><h2>📝 Text Detection</h2><p>Detect text regions in images</p></div>', unsafe_allow_html=True)
        
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
            st.experimental_rerun()

if __name__ == "__main__":
    main()
