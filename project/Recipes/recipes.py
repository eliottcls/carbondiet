import pandas as pd 
import os
import re
import warnings

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
    def __init__(self, name: str):
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
        # An ingredient is an instance of the class Ingredient
        self.ingredients = []
        # List of quantities (a quantity is a dict with 'quantity and 'unit')
        self.quantities = []
        # Score
        self.score_from_pefs = None

    def add_one_ingredient(self, ingredient_name: str, quantity_dict: dict):
        """
        Add one ingredient to the list of ingredients with its quantity

        Parameters
        ----------
        ingredient_name : str
        quantity_dict : dict
            Dictionnary with the 'quantity' and 'unit'
        """
        # here we should check that ingredient_name is e.g. in a fixed list of ingredients
        # to avoid problems with 'Boeuf' and 'boeuf' for instance
        # if the recipe has no ingredient with the name ingredient_name
        if ingredient_name not in [ing.name for ing in self.ingredients]:
            ingredient = Ingredient(ingredient_name)
            self.ingredients.append(ingredient)
            self.quantities.append(quantity_dict)
        # else nothing is done
        else:
            warnings.warn("This ingredient is already in the recipe. Nothing has been changed.")


    def average_from_recipes(self, recipe_list: list, weight_list = None):
        """
        Build a new recipe by averaging the recipes in recipe_list

        Parameters
        ----------
        recipe_list : list
            A list of recipes from the class Recipe
        weight_list : list
            A list giving the weight (between 0 and 1) of each recipe in the new recipe
        """
        self.ingredients = []
        self.quantities = []

        if weight_list == None: 
            # Recipe weights are all set equal to 1
            weight_list = [1] * len(recipe_list)

        # Loop on the recipes
        for recipe, weight in zip(recipe_list, weight_list):
            # Check there is no doublons in the recipe
            ing_names = [ing.name for ing in recipe.ingredients]
            assert len(ing_names)==len(set(ing_names))

            # Loop on the ingredients in each recipe
            for ingredient, quantity in zip(recipe.ingredients, recipe.quantities):
                # Normalize the quantity with the recipe weight
                weighted_quantity = quantity.copy()
                weighted_quantity['quantity'] = quantity['quantity'] * weight / len(recipe_list)

                # if the ingredient is not yet in the new recipe, add it
                if ingredient.name not in [ing.name for ing in self.ingredients]:
                    self.ingredients.append(ingredient)
                    self.quantities.append(weighted_quantity)
                # else 
                else:
                    # find the index of the ingredient in the ingredient list
                    idx = [ing.name for ing in self.ingredients].index(ingredient.name)
                    # check units are the same
                    assert self.quantities[idx]['unit']==weighted_quantity['unit']
                    # update the quantity of the ingredient
                    self.quantities[idx]['quantity'] += weighted_quantity['quantity']

    #
    #def average_from_nlp_predictions(self, nlp_results: list):



