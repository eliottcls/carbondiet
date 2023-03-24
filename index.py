import streamlit as st
import cv2
import numpy as np
from project.ComputerVision.ocr import OCR


st.set_page_config(page_title="CarbonDiet", page_icon="🥗", layout="wide", initial_sidebar_state="auto", menu_items=None)

def app():
    st.header("🥗 Welcome to CarbonDiet app")
    st.markdown(
        """
    This demo scans a menu and extract recipes
    """,
        unsafe_allow_html=True,
    )

    ocr = OCR()

    st.sidebar.write("Select a restaurant menu to analyse..")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the contents of the file using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Perform any image processing tasks on the image here
        
        with st.spinner(text='Text extraction in progress'):
            dict_results = ocr.extract_text_from_menu(img)
            st.success("Done")

        tab1, tab2, tab3 = st.tabs(["Box detection on menu", "Cleaned box detection on menu", "Extracted text"])
        # Display the original and processed images
        tab1.image(ocr.display_boxes(img), caption="Raw box detection", use_column_width=True)
        tab2.image(ocr.display_cleaned_boxes(img), caption="Cleaned box detection", use_column_width=True)
        tab3.json(dict_results)

app()