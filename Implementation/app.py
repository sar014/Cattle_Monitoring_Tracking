import streamlit as st
import cv2
import tempfile
import pandas as pd
from processor import CattleAnalyzer

st.set_page_config(page_title="Cattle Health Dashboard", layout="wide")

st.title("🐄 Real-Time Cattle Behavior & Distress Monitor")

# Sidebar for Model Loading
st.sidebar.header("Configuration")
video_file = st.sidebar.file_uploader("Upload Cow Video", type=['mp4', 'avi', 'mov'])

if video_file:
    # Initialize Analyzer
    analyzer = CattleAnalyzer(
        detect_model='Cow_Detect/best.pt', 
        pose_model='Pose_Detect/best.pt',
        rf_model='final_cow_behavior_modelV2.pkl',
        scaler='cow_scaler.pkl'
    )

    # Temporary file handling
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Dashboard Layout
    col1, col2 = st.columns([2, 1])
    frame_window = col1.image([])
    status_table = col2.empty()
    chart_placeholder = col2.empty()

    history_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, behaviors = analyzer.process_frame(frame)

        # Update Video Feed
        frame_window.image(processed_frame)

        # Update Dashboard Data
        if behaviors:
            df = pd.DataFrame(behaviors)
            status_table.table(df)
            history_log.extend([b['behavior'] for b in behaviors])
            
            # Show behavior distribution chart
            counts = pd.Series(history_log).value_counts()
            chart_placeholder.bar_chart(counts)

    cap.release()
else:
    st.info("Please upload a video file in the sidebar to begin analysis.")