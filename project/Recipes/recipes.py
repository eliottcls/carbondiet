import pandas as pd 
import os
import re

class Jow:
    def __init__(self):
        self.raw_df = pd.read_json('data/recipes/recipes_jow.json').transpose()
        self.df = self.preprocessing()


    def save(self, filename = "preprocessed_jow_df"):
        filepath = "data/recipes/" + filename + ".pkl"
        # !!! This should be modified !!!
        if not os.path.isfile(filepath):  
            self.df.to_pickle(filepath)
        else:
            print("This file already exists.")


    def preprocessing(self, filename = "preprocessed_jow_df"):
        '''
        Preprocess the raw dataframe read from 'recipes_jow.json'
        '''
        def remove_ligatures(transcript):
            REGEX_REPLACEMENTS = [(r"\u0153", "oe"), \
                                  (r"\u0152", "Oe")]
            
            for old, new in REGEX_REPLACEMENTS:
                transcript = re.sub(old, new, transcript, flags=re.IGNORECASE)
            return transcript

        def extract_ingredients(row):
            meaningless_jow_ingredients = ['1/10 bou.', '1/5 bou.', '1/50 bou.', '6 pinc.', \
                                           '1 càc', '1/4', '3 pinc.', '2 càs', '1 cm', '1/20 bou.', 
                                           '1 càs', '1/2 càc', 'Papier cuisson', 'Pics à brochette']
            
            ingredients = []
            for x in row:
                if len(x)>1:         
                    ingredient = x[-1]    # the ingredient is the last element of the list 'row'  
                    if ingredient not in meaningless_jow_ingredients:   # keep only meaningfull ingredients
                        ingredient = remove_ligatures(ingredient)
                        ingredients.append(ingredient) 

            #remove informations in parenthesis
            ingredients = [re.sub("[\(\[].*?[\)\]]", "", ingredient) for ingredient in ingredients]  
            # remove blank at the end of the string that remained when parenthesis have been removed
            ingredients = [ing[:-1] if ing[-1]==" " else ing for ing in ingredients]   

            return ingredients

        try:
            # Try to load the dataframe
            filepath = "data/recipes/" + filename + ".pkl"
            self.df = pd.read_pickle(filepath)

        except:
            # Add a column 'recipe_name' (copy of index)
            self.df = self.raw_df.copy()
            self.df['recipe_name'] = self.df.index
            # Rename the column 'ingredients'
            self.df = self.df.rename({'ingredients': 'ingredients_with_quantity'}, axis = 1)
            # Add a new column 'ingredients' for ingredients only (no quantity, informations in parenthesis are removed)
            self.df['ingredients'] = self.df['ingredients_with_quantity'].apply(extract_ingredients)
            # Add the column 'name_with_ingredients' by concatenating recipe names and ingredients
            self.df['name_with_ingredients'] = self.df.apply(lambda row: row['recipe_name'] \
                                                            + ", " + ', '.join(row['ingredients']), axis = 1)
            # Save the dataframe
            self.save(filename)
        
        return self.df
        

###################################################################################

class Ingredient:
    def __init__(self, name):
        # Ingredient name 
        self.name = name
        # List of corresponding Agribalyse ingredients
        self.agribalyse_ingredients = None
        # Mean PEF score (unit = mPt/kg)
        self.pef_score = None

class Recipe:
    def __init__(self):
        # Recipe name 
        self.name = None
        # List of ingredients
        self.ingredients = None
        # List of quantities (including units)
        self.quantities = None
        # Score
        self.score_from_pefs = None

    def add_one_ingredient(ingredient, quantity):
        """
        Add one ingredient to the list of ingredients

        Parameter
        ---------
        ingredient : instance of the class Ingredient
        """
        if ingredient not in self.ingredients:
            self.ingredients.append(ingredient)
            self.quantities.append(quantity)
        else:
            pass