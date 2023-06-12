import streamlit as st
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append("../")    # Add the path to the root directory (where we can find the folder project/)
from project.ComputerVision.ocr import OCR
import project.NLP.nlp as nlp
from project.Recipes.menu import Menu

# Add this line to avoid re-running the entire Python script from top to bottom
# any time something is updated on the screen (due to a widget for instance)
#@st.cache(suppress_st_warning=True)
def load_OCR_model():
    ocr = OCR()
    return ocr

 #@st.cache(suppress_st_warning=True)
def load_NLP_model():
    nlp_transformer = nlp.RecipeTransformer(db_name = "jow")
    return nlp_transformer



def app():
    st.set_page_config(page_title="CarbonDiet", page_icon="🥗", layout="wide", initial_sidebar_state="auto", menu_items=None)
    st.header("🥗 Welcome to CarbonDiet app")
    st.markdown(
        """
    This demo scans a menu and extract recipes
    """,
        unsafe_allow_html=True,
    )

    # Instantiate OCR and NLP models
    with st.spinner(text='Loading OCR model'):
        ocr = load_OCR_model()
        st.success("The OCR model is loaded.", icon="✅")
    with st.spinner(text='Loading NLP model'):
        nlp_transformer = load_NLP_model()
        st.success("The NLP model is loaded.", icon="✅")

    # Choose the menu in the sidebar
    st.sidebar.write("Select a restaurant menu to analyse..")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])



    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the contents of the file using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Instantiate the menu
        menu = Menu(uploaded_file.name)

        # Perform any image processing tasks on the image here
        with st.spinner(text='Text extraction in progress'):
            dict_results = ocr.extract_text_from_menu(img)
            menu.add_ocr_output(dict_results)
            menu.clean_ocr_output()
            st.success("Text extraction is done.", icon="✅")

       

        tab1, tab2, tab3, tab4 = st.tabs(["Box detection on menu", 
                                          "Cleaned box detection on menu", 
                                          "Extracted text",
                                          "NLP predictions"
                                          ])
        # Display the original image
        with tab1:
            st.image(ocr.display_boxes(img), caption="Raw box detection", use_column_width=True)

        # Display the processed image
        with tab2:
            st.image(ocr.display_cleaned_boxes(img), caption="Cleaned box detection", use_column_width=True)

        # Display the extracted text
        with tab3:
            menu_df = menu.clean_ocr_output_df
            st.dataframe(menu_df)

         
        with tab4:
            # Run NLP predictions without recipe averaging
            st.write("For each menu entry, the NLP model finds the `k` closest recipes in our recipe database.")
            top_k = st.number_input('Choose the `k` value', min_value = 1, max_value = 10)

            NLP_clicked = st.button("Run the NLP predictions")   
            if NLP_clicked:
                with st.spinner(text='NLP prediction in progress'):
                    query_names = menu.queries("Title and Ingredients")
                    nlp_results = nlp_transformer.predict(query_names, top_k = top_k)
                    st.success("NLP predictions are done.", icon="✅")

                menu_df = menu.add_nlp_predictions(nlp_results, average = False)
                st.dataframe(menu_df)

                # Average recipes
                st.write("For each menu entry, let's average the `k` closest recipes predicted by NLP \
                        and discard ingredients that contribute less than a `threshold` value to the recipe PEF score.")
                threshold = st.slider('Choose the `threshold` value', min_value = 0.0, max_value = 1.0, step = 0.01)

                avgNLP_clicked = st.button("Average the NLP predictions")  
                if avgNLP_clicked: 
                    menu_df = menu.add_nlp_predictions(nlp_results, average = True, threshold = threshold)
                    st.dataframe(menu_df)

app()