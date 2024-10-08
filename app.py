import streamlit as st
import PIL
import cv2
import numpy
import utils
import io
from camera_input_live import camera_input_live  # Importing camera_input_live

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
            visualized_image = utils.predict_image(frame, conf_threshold)
            st_frame.image(visualized_image, channels="BGR")

        else:
            camera.release()
            break

# New function to handle live camera input
def play_camera_input():
    camera_frames = camera_input_live()  # Assuming this function yields frames from the camera

    st_frame = st.empty()
    for frame in camera_frames:
        visualized_image = utils.predict_image(frame, conf_threshold)
        st_frame.image(visualized_image, channels="BGR")

# Page configuration with updated labels
st.set_page_config(
    page_title="Person and Face Detection",
    page_icon=":bust_in_silhouette:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Updated title
st.title("Person and Face Detection Project :bust_in_silhouette:")

# Sidebar label changes
st.sidebar.header("Detection Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

# Sidebar confidence threshold settings
st.sidebar.header("Confidence Threshold")
conf_threshold = float(st.sidebar.slider(
    "Select the Confidence Threshold", 10, 100, 20)) / 100

input = None
if source_radio == "IMAGE":
    st.sidebar.header("Upload Image")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold=conf_threshold)

        # Display the image with person/face detection
        st.image(visualized_image, channels="BGR")
        
        # Optionally show the original uploaded image
        st.image(uploaded_image)

    else:
        # Sample image as a placeholder
        st.image("assets/cover_image.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload Video")
    input = st.sidebar.file_uploader("Choose a video.", type=("mp4"))

    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"

        with open(temporary_location, "wb") as out:
            out.write(g.read())

        out.close()

    if temporary_location is not None:
        play_video(temporary_location)
        if st.button("Replay", type="primary"):
            pass

    else:
        st.video("assets/sample_video.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")

if source_radio == "WEBCAM":
    play_camera_input()  # Replacing the previous play_video(0) with play_camera_input
