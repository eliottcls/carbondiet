import pandas as pd
import re
import numpy as np
import project.Recipes.recipes as recipes


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
        
        
    def add_nlp_predictions(self, nlp_results, average = False, threshold = 0):
        """
        Update self.ocr_and_pred_df using NLP predictions.
        New columns are added to self.clean_ocr_output_df.

        Parameters
        ----------
        nlp_results : list
            A list of lists giving recipes predicted by NLP (output of RecipeTransformer.predict())
        average : boolean
            Whether or not to average the NLP predictions for each menu recipe
        threshold : float 
            A float between 0 and 1. 
            Only ingredients whose contribution to the PEF score is larger than threshold are kept.
            If threshold = 0 (default), all ingredients are kept.
        """
        if self.clean_ocr_output_df is not None:
            # If we do not average the NLP predictions
            if not average:
                recipes_db = recipes.Jow()

                # Don't like how it is written
                # To be modified by changing how recipe scores are calculted in class Recipe
                nlp_recipes_list = [[recipes_db.extract_recipe(title) for title in result['titles']] 
                                    for result in nlp_results]
                for recipe_list in nlp_recipes_list:
                    for recipe in recipe_list:
                        recipe.compute_score()

                menu_df = self.clean_ocr_output_df.copy()
                menu_df['recipe_title'] = [result['titles'] for result in nlp_results]
                menu_df['recipe_tag'] = [result['tags'] for result in nlp_results]
                menu_df['similarity_score'] = [result['similarity_scores'] for result in nlp_results]
                menu_df['similarity_rank'] = [result['similarity_ranks'] for result in nlp_results]
                menu_df['PEF_score'] = [[recipe.score_from_pefs for recipe in recipe_list] 
                                        for recipe_list in nlp_recipes_list]
                menu_df = menu_df.explode(['recipe_title', 'recipe_tag', 'similarity_score', \
                                        'similarity_rank', 'PEF_score'])

                self.ocr_and_pred_df = menu_df

            # if we average the NLP predictions into one single new recipe
            else:
                average_recipes = []
                for result in nlp_results:
                    average_recipe = recipes.Recipe()
                    try:
                        average_recipe.average_from_nlp_predictions(result, db_name = 'jow', threshold = threshold)
                    except:
                        average_recipe.add_one_ingredient("Cannot be calculated", dict(quantity = 0, unit = 'no unit'))
                    finally:
                        average_recipes.append(average_recipe)

                menu_df = self.clean_ocr_output_df.copy()
                menu_df['ingredients_with_quantities'] = [[(ing.name, np.round(qty['quantity'], 6), qty['unit']) 
                                                          for ing, qty in zip(average_recipe.ingredients, average_recipe.quantities)]
                                                          for average_recipe in average_recipes]
                menu_df['PEF_score'] = [average_recipe.score_from_pefs for average_recipe in average_recipes]

                self.ocr_and_pred_df = menu_df
    
        else:  # improve error handling
            raise ValueError("Update first clean_ocr_output_df")
        
        return self.ocr_and_pred_df


    

