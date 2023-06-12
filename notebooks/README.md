The folder `notebooks/` contains two subfolders `drafts/` and `tests/` described below.
## Drafts
The folder `notebooks/drafts` contains (quick & dirty) exploratory notebooks to analyse data, scrap data, or test various OCR and NLP methods. 

- `[Eliott] food_keywords.ipynb` : Short notebook to investigate the distribution of Agribalyse ingredients in the different categories ("Groupe') and subcategories ("Sous-groupe")
- `[Eliott] recipes_footprints.ipynb` : Notebook to match Jow ingredients with Agribalyse ingredients and store the carbon footprint
- `[Genevieve] ingredients_scores__explore.ipynb` : Exploratory notebook to get familiarized with JOW/Agribalyse databases and see how to match them
- `[Genevieve] ingredients_scores__process` : notebook which generates the output file "Jow_Agribalyse_ingredients_scores.xlsx" giving the list of Jow ingredients together with their corresponding Agribalyse ingredients and mean environmental scores
- `[Genevieve]find_closest_recipe_from_menu_by_hand` : notebook which generates the output files "MenuToJowRecipes_v1.xlsx", "MenuTo7tRecipes_v1.xlsx", ... Each menu recipe is associated by hand with one (or several) matching recipes in JOW and 7tomorrow databases.
- `[Genevieve]NLP_explore.ipynb` : Exploratory notebook to investigate how BERT NLP methods can be used to match menu recipes (from "MenuToJowRecipes_v1.xlsx") to JOW recipes. The different models are evaluated and compared knowing for each menu recipe the matching recipe(s) in JOW found by hand and stored in "MenuToJowRecipes_v1.xlsx".
- `[Genevieve]Translate_7t_en2fr.ipynb` : Short notebook to translate the 7tomorrow database in french.
- `[Genevieve]NLP_explore_en.ipynb` : Exploratory notebook to investigate how BERT NLP methods can be used to match menu recipes (from "MenuTo7tRecipes_v1.xlsx") to 7tomorrow recipes. The different models are evaluated and compared knowing for each menu recipe the matching recipe(s) in 7tomorrow found by hand and stored in "MenuTo7tRecipes_v1.xlsx".
- `[Genevieve]NLP_explore_build_recipe.ipynb` : Exploratory notebook to build 'new' (average) recipes from a bunch of JOW recipes pre-selected with (Bert) NLP  
- `[Genevieve]NLP_explore_jow+7t.ipynb` : Evaluation of the BERT NLP method on the 60 menu recipes from "MenuTo7tJOWRecipes_v1.json". The evaluation is done by matching with the JOW database only, the 7t database only, or the concatenation of both.
- `[Genevieve]recipes_clustering.ipynb` : Test of clustering methods on JOW recipes.  
- `[Ugo] recipes_quantity.ipynb` : Notebook to express quantities of JOW ingredients in grams.
- `[Genevieve]Add_one_col_quantity_df.ipynb` : Tiny notebook (following `[Ugo] recipes_quantity.ipynb`) adding a column for uniformized JOW ingredients i.e. without informations in parenthesis.
- `[Aurelien]marmiton_scraping.ipynb` : A notebook to scrap Marmiton recipes


## Tests
The folder `notebooks/tests` contains notebooks that are used to test the carbondiet library located in `project`.  
- **`test.ipynb` : main notebook to test the full process (CV+OCR+NLP -> predictions)**
- `test_recipes_lib.ipynb` : tests of the main functionalities of the module `project.Recipes.recipes`