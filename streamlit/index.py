import streamlit as st
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append("../")    # Add the path to the root directory (where we can find the folder project/)
from project.ComputerVision.ocr import OCR
import project.NLP.nlp as nlp
from project.Recipes.menu import Menu


# Add this decorator to avoid re-running the entire Python script from top to bottom
# any time something is updated on the screen (due to a widget for instance)
@st.cache_resource
def load_OCR_model():
    ocr = OCR()
    return ocr

@st.cache_resource
def load_NLP_model():
    nlp_transformer = nlp.RecipeTransformer(db_name = "jow")
    return nlp_transformer


@st.cache_data
def load_menu_image(file):
    # Read the contents of the file using OpenCV
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img


@st.cache_data
def init_menu(file):
    menu = Menu(file.name)
    return menu

# parameters with an underscore attached to the front (_menu and _ocr_model) are ignored for hashing.
# Even if they change, Streamlit will return the cached result if the value of 'menu_img' is unchanged.
@st.cache_data
def update_menu_with_extracted_text(_menu, menu_img, _ocr_model):
    # Extract text
    dict_results = _ocr_model.extract_text_from_menu(menu_img)
    # Add it to the menu
    _menu.add_ocr_output(dict_results)
    _menu.clean_ocr_output()

    return _menu

@st.cache_data
def compute_nlp_predictions(query_names, top_k, _nlp_model):
    nlp_results = _nlp_model.predict(query_names, top_k = top_k)
    return nlp_results




def app():
    st.set_page_config(page_title="CarbonDiet", page_icon="🥗", layout="wide", initial_sidebar_state="auto", menu_items=None)
    st.header("🥗 Welcome to CarbonDiet app")
    st.markdown(
        """
    This demo scans a menu, extract its meals and finds the closest recipes in our (JOW) recipe database.  
    Averaging over a bunch of matching recipes is possible.  
    The estimated environmental footprint of each recipe is returned.
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
        # Load the menu image
        img = load_menu_image(uploaded_file)

        # Instantiate the menu
        menu = init_menu(uploaded_file)

        # Extract text
        with st.spinner(text='Text extraction in progress'):
            menu = update_menu_with_extracted_text(menu, img, ocr)
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
            st.dataframe(menu_df.drop("Title and Ingredients", axis = 1), 
                         column_config = {"Title": "Category",
                                          "Ingredients": "Dish"})

         
        with tab4:
            # Initialize values in Session State
            if 'nlp_results' not in st.session_state:
                st.session_state['nlp_results'] = None

            # Run NLP predictions without recipe averaging
            st.write("For each menu entry, the NLP model finds the $k$ closest recipes in our recipe database.")
            col1, _ = st.columns([1, 5])
            top_k = col1.number_input('Choose the $k$ value', min_value = 1, max_value = 10)

            NLP_clicked = st.button("Run the NLP predictions")   
            if NLP_clicked:
                with st.spinner(text='NLP prediction in progress'):
                    query_names = menu.queries("Title and Ingredients")
                    nlp_results = compute_nlp_predictions(query_names, top_k, nlp_transformer)
                #st.success("NLP predictions are done.", icon="✅")

                with st.spinner(text='Adding NLP predictions to the dataframe'):
                    menu_df = menu.add_nlp_predictions(nlp_results, average = False)

                # Remove the recipe title from the recipe tag
                menu_df["recipe_tag"] = menu_df["recipe_tag"].apply(lambda s: ', '.join(s.split(', ')[1:]))
                # Display the dataframe
                st.dataframe(menu_df.drop("Title and Ingredients", axis = 1), 
                             column_config = {"Title": "Category",
                                              "Ingredients": "Dish",
                                              "recipe_title": "Closest recipe name",
                                              "recipe_tag": "Closest recipe ingredients",
                                              "similarity_score": "Similarity score",
                                              "similarity_rank": "Similarity rank",
                                              "PEF_score": "PEF score"})

                # Store nlp_results in the session state
                # to keep it even if the button is no more clicked
                st.session_state['nlp_results'] = nlp_results 

            # Average recipes
            st.write("For each menu entry, let's average the $k$ closest recipes predicted by NLP \
                    and discard ingredients that contribute less than a threshold value to the recipe PEF score.")
            col1, _ = st.columns([1, 3])
            threshold = col1.slider('Choose the threshold value (if 0, all ingredients are kept)', min_value = 0.0, max_value = 1.0, step = 0.01)

            avgNLP_clicked = st.button("Average the NLP predictions")  
            if avgNLP_clicked: 
                with st.spinner(text='Averaging NLP predictions'):
                    menu_df = menu.add_nlp_predictions(st.session_state['nlp_results'], average = True, threshold = threshold)

                menu_df['ingredients_with_quantities'] = menu_df['ingredients_with_quantities'].astype('str')
                st.dataframe(menu_df.drop("Title and Ingredients", axis = 1), 
                         column_config = {"Title": "Category",
                                          "Ingredients": "Dish",
                                          "ingredients_with_quantities": "Average recipe ingredients",
                                          "PEF_score": "PEF score"})


############
app()