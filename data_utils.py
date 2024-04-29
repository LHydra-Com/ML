#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold

#%%
def preprocess_data(data):
    # Handle missing values
    data = handle_missing_values(data)

    # Normalize or scale the features
    data = normalize_features(data)

    # Encode categorical variables
    data = encode_categorical_variables(data)

    return data

#%%
def handle_missing_values(data):
    # Handle missing values based on your requirements
    # For example, you can remove rows with missing values or fill them with a specific value
    data = data.dropna()  # Remove rows with missing values
    # data = data.fillna(0)  # Fill missing values with 0

    return data

#%%
def normalize_features(data):
    # Normalize or scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    numerical_columns = ['plays', 'popularity', 'duration_ms']  # Update with the available numerical columns in your dataset

    # Check if the numerical columns exist in the DataFrame
    available_columns = [col for col in numerical_columns if col in data.columns]

    if available_columns:
        data[available_columns] = scaler.fit_transform(data[available_columns])

    return data

#%%
def encode_categorical_variables(data):
    # Encode categorical variables using LabelEncoder
    categorical_columns = ['gender', 'country', 'artname']  # Update with the categorical columns in your dataset

    # Check if the categorical columns exist in the DataFrame
    available_columns = [col for col in categorical_columns if col in data.columns]

    label_encoder = LabelEncoder()
    for column in available_columns:
        data[column] = label_encoder.fit_transform(data[column])

    return data

#%%
def extract_user_item_interactions(data):
    # Extract user-item interactions from the dataset
    user_item_interactions = data.groupby(['usersha1', 'artname'])['plays'].sum().unstack(fill_value=0)
    return user_item_interactions

#%%
def extract_item_content_features(data):
    # Extract item content features from the dataset
    item_content_features = data.drop_duplicates(subset='artname')

    # Define the desired columns for item content features
    desired_columns = ['artname', 'genre', 'artist_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                       'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    # Check which desired columns are present in the DataFrame
    available_columns = [col for col in desired_columns if col in item_content_features.columns]

    # Select only the available columns
    item_content_features = item_content_features[available_columns]

    # Set 'artname' as the index
    item_content_features = item_content_features.set_index('artname')

    return item_content_features

#%%
def split_data(user_item_interactions, n_splits=5, random_state=None):
    # Get the user and item IDs
    user_ids = user_item_interactions.index.tolist()
    item_ids = user_item_interactions.columns.tolist()
#%%
    # Create a list of user-item pairs
    user_item_pairs = []
    for user_id in user_ids:
        for item_id in item_ids:
            user_item_pairs.append((user_id, item_id))
#%%
    # Create the interaction labels
    interaction_labels = [1 if user_item_interactions.loc[user_id, item_id] > 0 else 0
                          for user_id, item_id in user_item_pairs]

    # Initialize the KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize lists to store the train and test splits
    train_data_list = []
    test_data_list = []
#%%
    # Perform cross-validation splits
    for train_indices, test_indices in kf.split(user_item_pairs):
        # Get the train and test user-item pairs and interaction labels
        train_pairs = [user_item_pairs[i] for i in train_indices]
        test_pairs = [user_item_pairs[i] for i in test_indices]
        train_labels = [interaction_labels[i] for i in train_indices]
        test_labels = [interaction_labels[i] for i in test_indices]
#%%
        # Convert the train and test pairs into DataFrames
        train_data = pd.DataFrame(train_pairs, columns=['user_id', 'item_id'])
        train_data['interaction'] = train_labels
        test_data = pd.DataFrame(test_pairs, columns=['user_id', 'item_id'])
        test_data['interaction'] = test_labels
#%%
        # Append the train and test DataFrames to the respective lists
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    return train_data_list, test_data_list