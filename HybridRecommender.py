import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import ndcg_score, mean_reciprocal_rank, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import GridSearchCV

class HybridRecommender:
    def __init__(self, user_item_interactions, item_content_features):
        self.user_item_interactions = user_item_interactions
        self.item_content_features = item_content_features
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self.combined_similarity_matrix = None
        self.model = None

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

    def calculate_item_similarity(self):
        # Calculate item similarity based on content features
        self.item_similarity_matrix = cosine_similarity(self.item_content_features)

    def calculate_user_similarity(self):
        # Calculate user similarity based on user-item interactions
        self.user_similarity_matrix = cosine_similarity(self.user_item_interactions)

    def combine_similarity_matrices(self, alpha=0.5):
        # Combine item similarity and user similarity matrices
        self.combined_similarity_matrix = alpha * self.item_similarity_matrix + (1 - alpha) * self.user_similarity_matrix

    def train_model(self, n_neighbors=10):
        # Train the nearest neighbors model on the combined similarity matrix
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
        self.model.fit(self.combined_similarity_matrix)

    def tune_hyperparameters(self, param_grid, cv=5):
        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(self.combined_similarity_matrix)
        self.model = grid_search.best_estimator_

    def generate_recommendations(self, user_id, top_n=10):
        # Generate recommendations for a given user
        user_index = self.user_item_interactions.index.get_loc(user_id)
        user_vector = self.user_item_interactions.iloc[user_index].values.reshape(1, -1)

        # Find the nearest neighbors of the user
        distances, indices = self.model.kneighbors(user_vector)

        # Get the item indices of the nearest neighbors
        item_indices = indices.flatten()

        # Get the recommended items
        recommended_items = self.item_content_features.index[item_indices]

        return recommended_items[:top_n]

    def evaluate(self, test_data, top_n=10):
        # Evaluate the recommender system using the test data
        user_ids = test_data['user_id'].unique()

        ndcg_scores = []
        mrr_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        map_scores = []

        for user_id in user_ids:
            # Get the true items for the user
            true_items = test_data[test_data['user_id'] == user_id]['item_id'].tolist()

            # Generate recommendations for the user
            recommended_items = self.generate_recommendations(user_id, top_n)

            # Calculate evaluation metrics
            ndcg_scores.append(ndcg_score([true_items], [recommended_items]))
            mrr_scores.append(mean_reciprocal_rank([true_items], [recommended_items]))
            precision_scores.append(precision_score(true_items, recommended_items, average='micro'))
            recall_scores.append(recall_score(true_items, recommended_items, average='micro'))
            f1_scores.append(f1_score(true_items, recommended_items, average='micro'))
            map_scores.append(average_precision_score([true_items], [recommended_items]))

        # Calculate average evaluation metrics
        avg_ndcg = np.mean(ndcg_scores)
        avg_mrr = np.mean(mrr_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        avg_map = np.mean(map_scores)

        return {
            'NDCG': avg_ndcg,
            'MRR': avg_mrr,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1-score': avg_f1,
            'MAP': avg_map
        }

    def get_trending_items(self, top_n=10):
        # Get the trending items based on overall popularity
        item_popularity = self.user_item_interactions.sum(axis=0)
        trending_items = item_popularity.sort_values(ascending=False).