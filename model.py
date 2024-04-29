#%%
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

#%%
class HybridRecommender:
    def __init__(self, user_item_interactions, item_content_features):
        self.user_item_interactions = user_item_interactions
        self.item_content_features = item_content_features
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self.combined_similarity_matrix = None
        self.model = None
#%%
    def preprocess_data(self):
        # Normalize the user-item interaction data
        scaler = MinMaxScaler()
        self.user_item_interactions = pd.DataFrame(scaler.fit_transform(self.user_item_interactions),
                                                   columns=self.user_item_interactions.columns,
                                                   index=self.user_item_interactions.index)

        # Normalize the item content feature data
        self.item_content_features = pd.DataFrame(scaler.fit_transform(self.item_content_features),
                                                  columns=self.item_content_features.columns,
                                                  index=self.item_content_features.index)
#%%
    def calculate_item_similarity(self):
        # Calculate item similarity based on content features
        self.item_similarity_matrix = cosine_similarity(self.item_content_features)
#%%
    def calculate_user_similarity(self):
        # Calculate user similarity based on user-item interactions
        self.user_similarity_matrix = cosine_similarity(self.user_item_interactions)
#%%
    def combine_similarity_matrices(self, alpha=0.5):
        # Combine item similarity and user similarity matrices
        self.combined_similarity_matrix = alpha * self.item_similarity_matrix + (1 - alpha) * self.user_similarity_matrix
#%%
    def train_model(self):
        # Train the nearest neighbors model on the combined similarity matrix
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.combined_similarity_matrix)
#%%
    def train(self, train_data):
        # Preprocess the data
        self.preprocess_data()

        # Calculate item similarity matrix
        self.calculate_item_similarity()

        # Calculate user similarity matrix
        self.calculate_user_similarity()

        # Combine similarity matrices
        self.combine_similarity_matrices()

        # Train the nearest neighbors model
        self.train_model()
#%%
    def generate_recommendations(self, user_id, top_n=10):
        # Generate recommendations for a given user
        user_index = self.user_item_interactions.index.get_loc(user_id)
        user_vector = self.user_item_interactions.iloc[user_index].values.reshape(1, -1)

        # Find the nearest neighbors of the user
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=top_n+1)

        # Get the item indices of the nearest neighbors (excluding the user itself)
        item_indices = indices.flatten()[1:]

        # Get the recommended items
        recommended_items = self.item_content_features.index[item_indices]

        return recommended_items
#%%
    def evaluate(self, test_data, top_n=10):
        # Get the user IDs from the test data
        user_ids = test_data['user_id'].unique()

        # Initialize lists to store the true and predicted labels
        y_true = []
        y_pred = []

        # Iterate over each user in the test data
        for user_id in user_ids:
            # Get the true items for the user
            true_items = test_data[test_data['user_id'] == user_id]['item_id'].tolist()

            # Generate recommendations for the user
            recommended_items = self.generate_recommendations(user_id, top_n)

            # Create the true labels (1 if the item is in the true items, 0 otherwise)
            true_labels = [1 if item in true_items else 0 for item in recommended_items]

            # Create the predicted labels (1 for recommended items)
            predicted_labels = [1] * len(recommended_items)

            # Append the true and predicted labels to the lists
            y_true.extend(true_labels)
            y_pred.extend(predicted_labels)
#%%
        # Calculate the evaluation metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        map_score = average_precision_score(y_true, y_pred)

        # Return the evaluation metrics as a dictionary
        return {
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'MAP': map_score
        }
#%%
    def get_trending_songs(self, top_n=10):
        # Get the trending songs based on overall popularity
        trending_songs = self.user_item_interactions.sum().sort_values(ascending=False).head(top_n).index
        return trending_songs