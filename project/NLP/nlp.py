from sentence_transformers import SentenceTransformer, util
import project.Recipes.recipes as recipes
import pandas as pd
import numpy as np
import os

class RecipeTransformer:
    def __init__(self, 
                 model_name = "dangvantuan/sentence-camembert-large",
                 db_name = "jow"):
        
        # Initialization of the NLP model
        print("### LOADING THE MODEL ###")
        assert model_name in ["dangvantuan/sentence-camembert-large"]  # add more models
        self.model =  SentenceTransformer(model_name)
        print("The model is loaded.")

        # Initialization of the recipe database
        print("### LOADING THE DATABASE ###")
        self.database_name = None
        self.database_titles = None
        self.database_tags = None
        self.database_title_embeddings = None
        self.database_tag_embeddings = None
        self.load_database(db_name = db_name)
        print("The database is loaded.")


    def attach_database(self, db_name = "jow"):
        assert db_name in ['jow']    #add more recipe databases

        # Name
        self.database_name = db_name
        # Load and preprocess the JOW database
        recipes_db = recipes.Jow()
        recipes_db = recipes_db.df
        # List of recipe titles
        self.database_titles = list(recipes_db["recipe_name"].values)
        # List of recipe tags
        self.database_tags = list(recipes_db["name_with_ingredients"].values)
        # Embeddings
        self.database_title_embeddings = self.compute_embeddings(self.database_titles)
        self.database_tag_embeddings = self.compute_embeddings(self.database_tags)


    def compute_embeddings(self, recipe_tags):
        """ 
        Computes the recipe embeddings

        Parameters
        ----------
        recipe_tags : list
            List of recipes characterized by their tags
        Returns
        -------
        recipe_embeddings : list
            List of embedding vectors
        """
        recipe_embeddings = self.model.encode(recipe_tags, convert_to_tensor=True)

        return recipe_embeddings
    

    def save_database(self, filename):
        database = dict(
            name = self.database_name,
            titles = self.database_titles,
            tags = self.database_tags,
            title_embeddings = self.database_title_embeddings,
            tag_embeddings = self.database_tag_embeddings
        )

        filepath = "data/recipes/" + filename + ".npy"
        # !!! This should be modified !!!
        if not os.path.isfile(filepath):  
            np.save(filepath, database)
        else:
            print("This file already exists.")



    def load_database(self, db_name = "jow"):
        filename = db_name + "_embedding"

        try:
            # Load the database if the file already exists
            filepath = "data/recipes/" + filename + ".npy"
            database = np.load(filepath, allow_pickle = True).reshape(-1)[0]
            
            self.database_name = database['name']
            self.database_titles = database['titles']
            self.database_tags = database['tags']
            self.database_title_embeddings = database['title_embeddings']
            self.database_tag_embeddings = database['tag_embeddings']
        
        except:
            # Attach the database and save it
            self.attach_database(db_name = db_name)
            self.save_database(filename)



    def extract_from_database(self, tag_name = "name_with_ingredients"):
        '''
        Returns the corpus of sentences used for the sentence similarity task
        together with their embedding vectors

        Parameters
        ----------
        tag_name : str
            To choose which corpus to use
        '''
        assert tag_name in ["name_with_ingredients", "recipe_name"]

        # Use the recipe titles as corpus
        if tag_name == "recipe_name":
            database_tags = self.database_titles
            database_embeddings = self.database_title_embeddings

        # Use the list of recipe titles concatenated with their ingredients as corpus
        if tag_name == "name_with_ingredients":
            database_tags = self.database_tags
            database_embeddings = self.database_tag_embeddings

        return database_tags, database_embeddings


    def find_k_closest_recipes(self, query_embedding, database_embeddings, top_k = 20):
        """
        Returns the hits in the database corresponding to the query,
        whose similarity scores are among the top_k biggest scores 
        """
        hits = util.semantic_search(query_embedding, database_embeddings, top_k = top_k)    #find the top_k closest database recipes from query
        hits = hits[0]      #Get the hits for the first query (here only one query is given)

        return hits
    
    # Be careful, not tested here
    def find_recipes_with_score_threshold(self, query_embedding, database_embeddings, radius = 0.1):
        """
        Returns the hits in the database corresponding to the query,
        whose similarity scores are larger than max_score - radius
        """
        hits = self.find_k_closest_recipes(query_embedding, database_embeddings, 
                                           top_k = database_embeddings.shape[0])

        # Keep only the recipes whose similarity scores are higher than max_score - radius
        best_hits = []
        for k, hit in enumerate(hits):
            max_score = hits[0]['score']  # hits is stored by descending scores
            if hit['score'] > max_score - radius:
                best_hits.append(hit)

        return best_hits
    

    def predict_from_single_query(self, query_embedding, database_embeddings, method = 'top', top_k = 3, radius = 0.1):
        assert method in ['top', 'threshold']

        if method=='top':
            hits = self.find_k_closest_recipes(query_embedding, database_embeddings, top_k = top_k)
        if method=='threshold':
            hits = self.find_recipes_with_score_threshold(query_embedding, database_embeddings, radius = radius)   

        return hits 
    

    def predict(self, query_names, tag_name = "name_with_ingredients", method = 'top', top_k = 3, radius = 0.1):
        """
        Returns the NLP predictions for the menu queries listed in query_names

        Parameters
        ----------
        query_names : list of str
            List of menu queries
        tag_name : str
            String to select the NLP corpus for sentence similarity
        method : str
            Method used for the NLP (sentence similarity) task
            Either "top" or "threshold"
        top_k : integer
            Parameter if method == "top
        radius : float
            Parameter if method == "threshold"
        Returns
        -------
        results_list : list of dict
            Each element of the list is a dictionnary giving the NLP results
            corresponding to one menu query
        """
        database_tags, database_embeddings = self.extract_from_database(tag_name = tag_name)

        results_list = []
        for query_name in query_names:
            query_embedding = self.compute_embeddings(query_name)
            hits = self.predict_from_single_query(query_embedding, database_embeddings, 
                                                  method = method, top_k = top_k, radius = radius)

            query_dict = {}
            query_dict['tags'] = [database_tags[hit['corpus_id']] for hit in hits]
            query_dict['titles'] = [self.database_titles[hit['corpus_id']] for hit in hits]
            query_dict['similarity_scores'] = [hit['score'] for hit in hits]
            query_dict['similarity_ranks'] = [k + 1 for k in range(len(hits))]

            results_list.append(query_dict)
            
        return results_list




    


    

