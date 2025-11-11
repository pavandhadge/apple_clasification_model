import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model import load_model, predict
from utils import prob_bad_from_preds, sample_video_frames, smoothed_prediction


def main_ui(tf_available):
    st.set_page_config(page_title="Apple Health Detector", layout="wide", initial_sidebar_state="collapsed")

    st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                div.stButton > button {
                    height: 3em;
                }
        </style>
        """, unsafe_allow_html=True)

    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    st.title("🍎 Apple Health Detector")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Instructions")
        st.write("1. Select the type of file you want to analyze (Image or Video).")
        st.write("2. Upload the file.")
        st.write("3. Click the 'Analyze' button to see the prediction.")

        st.header("Upload")
        model = load_model("model/apple_model.h5")

        if model:
            upload_type = st.radio("", ("Image", "Video"), horizontal=True)

            if upload_type == "Image":
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            else:
                uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

            if uploaded_file:
                if st.session_state.uploaded_file != uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.prediction = None

                if st.button("Analyze", use_container_width=True):
                    with st.spinner('Analyzing...'):
                        if upload_type == "Image":
                            image = Image.open(uploaded_file).convert("RGB")
                            img_np = np.array(image)
                            pred = predict(model, img_np)
                            st.session_state.prediction = prob_bad_from_preds(pred)
                        else:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1].lower()) as tfile:
                                tfile.write(uploaded_file.read())
                                tmp_path = tfile.name

                            cap = cv2.VideoCapture(tmp_path)
                            frames, _ = sample_video_frames(cap)
                            cap.release()

                            if len(frames) > 0:
                                frame_results = []
                                for fr in frames:
                                    pred = predict(model, fr)
                                    smooth_class = smoothed_prediction(pred[0])
                                    frame_results.append(smooth_class)

                                final_class = np.bincount(frame_results).argmax()
                                st.session_state.prediction = final_class
                            else:
                                st.error("Could not read frames from the video.")

    with col2:
        st.header("Result")
        if st.session_state.prediction is not None:
            if st.session_state.prediction >= 0.5:
                st.error("### Prediction: BAD Apple")
                st.write("The model has detected signs of spoilage, such as bruises, discoloration, or mold.")
            else:
                st.success("### Prediction: GOOD Apple")
                st.write("The model has not detected any significant signs of spoilage.")

            st.metric("Confidence Score", f"{st.session_state.prediction:.2f}")

            with st.expander("Show Preview"):
                if st.session_state.uploaded_file.type.startswith("image"):
                    st.image(st.session_state.uploaded_file, use_column_width=True)
                else:
                    st.video(st.session_state.uploaded_file)
        else:
            st.info("Upload a file and click 'Analyze' to see the result.")
