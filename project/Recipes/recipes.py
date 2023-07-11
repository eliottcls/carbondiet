import pandas as pd 
import numpy as np
import os
import re
import warnings

# path of the root directory
DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )  


class Jow:
    def __init__(self):
        filepath = os.path.join(DIR, "data/recipes", "recipes_jow.json")
        self.raw_df = pd.read_json(filepath).transpose()
        self.df = self.preprocessing()


    def save(self, filename = "preprocessed_jow_df"):
        filepath = os.path.join(DIR, "data/recipes", filename + ".pkl")
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
            quantities = []
            for x in row:
                if len(x)>1:         
                    ingredient = x[-1]    # the ingredient is the last element of the list 'row'  
                    quantity = x[-2]      # x[0] might be "Faculatif" and not the quantity
                    if ingredient not in meaningless_jow_ingredients:   # keep only meaningfull ingredients
                        ingredient = remove_ligatures(ingredient)
                        ingredients.append(ingredient) 
                        quantities.append(quantity) 

            #remove informations in parenthesis
            ingredients = [re.sub("[\(\[].*?[\)\]]", "", ingredient) for ingredient in ingredients]  
            # remove blank at the end of the string that remained when parenthesis have been removed
            ingredients = [ing[:-1] if ing[-1]==" " else ing for ing in ingredients]   

            return [ingredients, quantities]

        try:
            # Try to load the dataframe
            filepath = os.path.join(DIR, "data/recipes", filename + ".pkl")
            self.df = pd.read_pickle(filepath)

        except:
            # Add a column 'recipe_name' (copy of index)
            self.df = self.raw_df.copy()
            self.df['recipe_name'] = self.df.index
            # Rename the column 'ingredients'
            self.df = self.df.rename({'ingredients': 'ingredients_with_quantity'}, axis = 1)
            # Add a new column for ingredients without informations in parenthesis and irrelevant ingredients
            self.df['simple_ingredients_with_quantity'] = self.df['ingredients_with_quantity'].apply(extract_ingredients)
            # Add a new column 'ingredients' for ingredients only (no quantity, informations in parenthesis are removed)
            self.df['ingredients'] = self.df['ingredients_with_quantity'].apply(lambda x: extract_ingredients(x)[0])
            # Add the column 'name_with_ingredients' by concatenating recipe names and ingredients
            self.df['name_with_ingredients'] = self.df.apply(lambda row: row['recipe_name'] \
                                                            + ", " + ', '.join(row['ingredients']), axis = 1)
            # Save the dataframe
            self.save(filepath)
        
        return self.df
    
    
    def extract_recipe(self, recipe_title: str):
        """ 
        Extract the recipe from the Jow database with the title 'recipe_title'

        Parameters
        ----------
        recipe_title : str
            Title of the Jow recipe to be extracted
        Returns
        -------
        recipe : instance of the class Recipe
            The extracted recipe
        """

        # Utility function to convert a string n representing a number into a float
        def floatenize(n):  
            try:
                n = float(n)
            except ValueError:
                num, denom = n.split('/')
                n = float(num) / float(denom)
            return n

        # Instantiate an empty recipe
        recipe = Recipe()
        # Get the ingredients + quantities of the Jow recipe with title 'recipe_title'
        ingredients_quantities = self.df.loc[recipe_title]['simple_ingredients_with_quantity']
        # Add them to the recipe
        for ingredient_name, quantity in zip(*ingredients_quantities):
            # Build the quantity dict of each ingredient
            if len(quantity.split(" "))==1:
                nb = floatenize(quantity.split(" ")[0])
                unit = "unitaire"
            else:
                nb, unit = quantity.split(" ")
                nb = floatenize(nb)

            quantity_dict = dict(quantity = nb, unit = unit)

            # Add the ingredient with the correct quantity
            recipe.add_one_ingredient(ingredient_name, quantity_dict)

        return recipe

       
    

        

###################################################################################

class Ingredient:
    # To be completed
    # Not sure it is good to keep agribalyse_ingredients and pef_score as attributes
    # They could be calculated with methods.
    def __init__(self, name: str):
        # Ingredient name 
        self.name = name
        # List of corresponding Agribalyse ingredients
        self.agribalyse_ingredients = None
        # Mean PEF score (unit = mPt/kg)
        self.pef_score = None

    def add_agribalyse_infos(self):
        """
        Update self.agribalyse_ingredients and self.pef_score
        using the table matching JOW and Agribalyse ingredients.
        !! To be generalized outside JOW !!
        """
        filepath = os.path.join(DIR, "data/recipes", "Jow_Agribalyse_ingredients_scores.json")
        # Not well written because we have to read the file for each ingredient -> to be corrected
        df = pd.read_json(filepath)
        # Keep only the relevant line
        df = df[df['JOW ingredients (simple, fr)']==self.name]
        # Get agribalyse ingredients
        agribalyse_ingredients = list(df['AGB ingredients (simple, fr)'].values)

        # If there is no matching agribalyse ingredients
        if len(agribalyse_ingredients)==0 or agribalyse_ingredients[0]=='no match': 
            warnings.warn("The ingredient '" + self.name + "' cannot be matched with any of the Agribalyse ingredients. \
                           By default, it is associated with the global ingredient category.")
            self.agribalyse_ingredients = 'no match'
        # else update the attributes
        else:
            self.agribalyse_ingredients = agribalyse_ingredients[0]

        # Mean PEF Score
        self.pef_score = df['Mean PEF'].values[0]



###################################################################################

class Recipe:
    def __init__(self):
        # Recipe name 
        self.name = None
        # List of ingredients
        # An ingredient is an instance of the class Ingredient
        self.ingredients = []
        # List of quantities (a quantity is a dict with 'quantity and 'unit')
        self.quantities = []
        # List of contributions of each ingredient to the score
        self.score_contributions = []
        # Score
        self.score_from_pefs = None

        # Check there is one and only one quantity for each ingredient
        assert(len(self.ingredients)==len(self.quantities))

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

    def convert_quantities_in_kg(self):
        """
        Update self.quantities by expressing all quantities in kg
        (except for couples (self.ingredients[i], self.quantities[i]['unit'])
        that cannot be found in the conversion table)
        """
        # Be careful that this file gives the conversion table for JOW ingredients only
        filepath = os.path.join(DIR, "data/recipes", "95_perc_kg_unit_ingredients_v2.csv")
        qty_df = pd.read_csv(filepath)

        for idx, ingredient in enumerate(self.ingredients):
            # Do nothing if the quantity is already expressed in kg
            if self.quantities[idx]['unit']=='kg': 
                continue

            # Find matching entries in the quantity conversion table qty_df
            select_condition = ((qty_df['simple_ingredient']==ingredient.name.lower()) 
                                & (qty_df['name_unit']==self.quantities[idx]['unit']))
            unit_in_kg = qty_df[select_condition]['unit_kg'].values.tolist()

            # if there is at least one matching entry in qty_df for (ingredient, quantity['unit])
            if len(unit_in_kg)!=0:
                # compute the mean value of unit_in_kg if there are several matching entries
                unit_in_kg = np.array(unit_in_kg).mean()
                quantity = unit_in_kg * self.quantities[idx]['quantity']
                # update the quantity dictionnary
                self.quantities[idx] = dict(quantity = quantity, unit = 'kg')
            # if not, raise a warning and keep unchanged the quantity dictionnary
            else:
                warnings.warn("The ingredient '" + ingredient.name + "' with unit '" + self.quantities[idx]['unit'] 
                              + "' has no matching entry in the quantity conversion table.")


    def compute_score(self):
        """
        Update self.score_from_pefs and self.score_contributions
        """
        # Convert quantities in kg
        self.convert_quantities_in_kg()
        # Could use try / except here but for now, better to keep the code running even if one quantity is not in kg
        # If one quantity is not expressed in kg, the score cannot be calculated
        if not all([quantity['unit']=='kg' for quantity in self.quantities]):
            warnings.warn("All quantities must be expressed in kg. The recipe score cannot be calculated")
            self.score_from_pefs = np.nan
            self.score_contributions = [np.nan for i in range(len(self.ingredients))]
        # else 
        else:
            score_ingredient_contributions = []
            for ingredient, quantity in zip(self.ingredients, self.quantities):
                ingredient.add_agribalyse_infos()
                # If the pef score of the ingredient cannot be calculated 
                # (because it cannot be matched with one of Agribalyse ingredients)
                # set the score to np.nan
                if np.isnan(ingredient.pef_score):
                    ingredient_pef_score = np.nan
                # else compute the score weighted by the quantity
                else:
                    ingredient_pef_score = ingredient.pef_score * quantity['quantity']
                score_ingredient_contributions.append(ingredient_pef_score)

            if not np.isnan(score_ingredient_contributions).any():
                self.score_from_pefs = sum(score_ingredient_contributions)
                self.score_contributions = score_ingredient_contributions / self.score_from_pefs
            else:
                self.score_from_pefs = np.nan
                self.score_contributions = [np.nan for i in range(len(self.ingredients))]



    def average_from_recipes(self, recipe_list: list, weight_list = None, threshold = 0):
        """
        Build a new recipe by averaging the recipes in recipe_list

        Parameters
        ----------
        recipe_list : list
            A list of recipes from the class Recipe
        weight_list : list
            A list giving the weight (between 0 and 1) of each recipe in the new recipe
        threshold : float 
            A float between 0 and 1. 
            Only ingredients whose contribution to the PEF score is larger than threshold are kept.
            If threshold = 0 (default), all ingredients are kept.
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

            # Convert quantities in kg
            recipe.convert_quantities_in_kg()

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

        # Remove ingredients whose contributions to the score is lower than threshold
        assert 0<=threshold<=1
        self.compute_score()
        if not np.isnan(self.score_from_pefs):
            self.ingredients = list(np.array(self.ingredients)[np.array(self.score_contributions) > threshold])
            self.quantities = list(np.array(self.quantities)[np.array(self.score_contributions) > threshold])
        



        

    
    def average_from_nlp_predictions(self, nlp_results: list, db_name = 'jow', threshold = 0):
        """
        Build a new recipe by averaging the recipes in nlp_results

        Parameters
        ----------
        nlp_results : list
            A list giving recipes predicted by NLP 
            (a sublist of the list returned by RecipeTransformer.predict())
        db_name : str
            -> TO BE MODIFIED, SHOULD BE INCLUDED IN NLP_RESULTS
            The name of recipe database used for NLP predictions
        threshold : float 
            A float between 0 and 1. 
            Only ingredients whose contribution to the PEF score is larger than threshold are kept.
            If threshold = 0 (default), all ingredients are kept.
        """

        assert db_name in ['jow']      #add more recipe databases

        # Not optimized here (no need to reload the recipes_db for each recipe)
        recipes_db = Jow()

        recipe_list = [recipes_db.extract_recipe(title) for title in nlp_results['titles']]
        recipe_scores = nlp_results['similarity_scores']
        recipe_weigths = list(np.array(recipe_scores) / max(recipe_scores))

        self.average_from_recipes(recipe_list, weight_list = recipe_weigths, threshold = threshold)




