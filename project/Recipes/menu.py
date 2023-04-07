import pandas as pd
import re


class Menu:
    def __init__(self, image_filename):
        self.image_filename = image_filename
        self.raw_ocr_output_dict = None
        self.clean_ocr_output_df = None
        self.ocr_and_pred_df = None

    def add_ocr_output(self, raw_ocr_output_dict):
        self.raw_ocr_output_dict = raw_ocr_output_dict

    def clean_ocr_output(self):
        """
        Converts self.raw_ocr_output_dict to a DataFrame
        and performs several cleaning steps to make it suitable for the NLP task
        """
        if self.raw_ocr_output_dict is not None:
            # Create a dataframe from ocr_output
            self.clean_ocr_output_df = pd.DataFrame.from_dict(self.raw_ocr_output_dict, orient = 'index')

            # Transform to get one line per recipe
            self.clean_ocr_output_df = self.clean_ocr_output_df.stack()
            self.clean_ocr_output_df = self.clean_ocr_output_df.rename_axis(['Title', 'subgroup_id'])
            self.clean_ocr_output_df = self.clean_ocr_output_df.rename('Ingredients')
            self.clean_ocr_output_df = self.clean_ocr_output_df.reset_index()
            self.clean_ocr_output_df = self.clean_ocr_output_df[['Title', 'Ingredients']]

            # Remove prives
            def remove_prices(menu_string):
                # Prices like 3.20€ or 3,20 EUR or ...17€ etc...
                price_regex1 = r"(\s|\.|-)*[0-9]+[,.]?[0-9]*\s*(€|EUR|EUROS|EURO|eur|euros|euro)+"
                # Prices like €3.20€ or EUR 3,20 EUR or €17 etc...
                price_regex2 = r"(€|EUR|EUROS|EURO|eur|euros|euro)+\s*[0-9]+[,.]?[0-9]*\s*"
                # Prices as numbers at the end of the string
                price_regex3 = r"(\s|\.|-)*[0-9]+[,.]?[0-9]*\s*$"
                for price_regex in [price_regex1, price_regex2, price_regex3]:
                    menu_string = re.sub(price_regex, "", menu_string)
                    
                return menu_string

            self.clean_ocr_output_df = self.clean_ocr_output_df.applymap(remove_prices)
            
            # Add a column concatenating the titles and the ingredients
            self.clean_ocr_output_df['Title and Ingredients'] = self.clean_ocr_output_df.apply(lambda row: row['Title'] + ", " + row['Ingredients'], axis = 1)

        else:  # improve error handling
            raise ValueError("Update first the attribute raw_ocr_output_dict with the method add_ocr_output")
        

    def queries(self, tag = "Title and Ingredients"):
        '''
        Returns the list of menu queries (for the NLP task) characterized by the tag
        '''
        assert tag in ["Title", "Ingredients", "Title and Ingredients"]

        if self.clean_ocr_output_df is None:
            self.clean_ocr_output()
        
        queries = list(self.clean_ocr_output_df[tag].values)

        return queries
        
        
    def add_nlp_predictions(self, nlp_results):
        """
        Add columns to the dataframe self.clean_ocr_output_df
        with NLP predictions and scores contained in nlp_results
        """
        if self.clean_ocr_output_df is not None:
            menu_df = self.clean_ocr_output_df.copy()
            menu_df['recipe_title'] = [result['titles'] for result in nlp_results]
            menu_df['recipe_tag'] = [result['tags'] for result in nlp_results]
            menu_df['similarity_score'] = [result['similarity_scores'] for result in nlp_results]
            menu_df['similarity_rank'] = [result['similarity_ranks'] for result in nlp_results]
            menu_df = menu_df.explode(['recipe_title', 'recipe_tag', 'similarity_score', 'similarity_rank'])

            self.ocr_and_pred_df = menu_df
    
        else:  # improve error handling
            raise ValueError("Update first clean_ocr_output_df")
        
        return self.ocr_and_pred_df


    

