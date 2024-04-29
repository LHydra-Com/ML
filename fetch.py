#%%
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
#%%
# Replace with your Spotify API credentials
client_id = '2bf3b69bac7441e785aa65e91a609cfe'
client_secret = 'b4ab1d18983542dca52e963258322b58'
#%%
# Authenticate with the Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
#%%
# Load the dataset
dataset = pd.read_csv('dataset.csv')
#%%
# Group users by demography
user_groups = dataset.groupby(['gender', 'age', 'country', 'signup_year'])
#%%
# Create an empty list to store the combined data
combined_data = []
#%%
# Iterate over each user group
for group_keys, group in user_groups:
    # Unpack the group keys (gender, age, country, signup_year)
    gender, age, country, signup_year = group_keys

    # Get the list of user IDs for this group
    user_ids = group['usersha1'].unique().tolist()

    # Fetch the top 10 artists for this group using the Spotify API
    top_artists = []
    for user_id in user_ids:
        try:
            # Get the user's recently played tracks
            recently_played = sp.current_user_recently_played(limit=50)
            recently_played_tracks = [item['track']['artists'][0]['name'] for item in recently_played['items']]

            # Count the occurrences of each artist
            artist_counts = dict()
            for artist in recently_played_tracks:
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

            # Sort the artists by their counts and get the top 10
            top_artists.extend([artist for artist, count in sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:10]])

            # Add a delay to avoid rate limiting by the Spotify API
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching data for user {user_id}: {e}")
#%%
    # Remove duplicates from the top artists list
    top_artists = list(set(top_artists))
#%%
    # Add the group information and top artists to the combined data
    for artist in top_artists:
        combined_data.append([gender, age, country, signup_year, artist])
#%%
# Create a DataFrame from the combined data
combined_df = pd.DataFrame(combined_data, columns=['gender', 'age', 'country', 'signup_year', 'top_artist'])
#%%
# Save the combined dataset to a CSV file
combined_df.to_csv('current_dataset.csv', index=False)
# %%
