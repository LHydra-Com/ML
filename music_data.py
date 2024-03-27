import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

class Music:
    def __init__(self):
        self.sp = self.initialize_spotipy()
        self.api_key = '4e4d5f65c4ccd249b6fb8b3f7b761eef'

    def initialize_spotipy(self):
        client_id = '4187992fdb764829b6b2ce20718027c0'
        client_secret = '4adc98b676ed40e1b43c521b355ef809'
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    def get_user_listening_history(self, user_id):
        # Retrieve user listening history from Spotify API
        results = self.sp.current_user_recently_played(limit=50)
        listening_history = []
        for item in results['items']:
            track = item['track']
            listening_history.append({
                'user_id': user_id,
                'track_id': track['id'],
                'artist_id': track['artists'][0]['id'],
                'timestamp': item['played_at']
            })
        return listening_history

    def get_track_audio_features(self, track_id):
        # Retrieve track audio features from Spotify API
        audio_features = self.sp.audio_features([track_id])[0]
        return {
            'track_id': track_id,
            'danceability': audio_features['danceability'],
            'energy': audio_features['energy'],
            'key': audio_features['key'],
            'loudness': audio_features['loudness'],
            'mode': audio_features['mode'],
            'speechiness': audio_features['speechiness'],
            'acousticness': audio_features['acousticness'],
            'instrumentalness': audio_features['instrumentalness'],
            'liveness': audio_features['liveness'],
            'valence': audio_features['valence'],
            'tempo': audio_features['tempo']
        }

    def get_artist_metadata(self, artist_id):
        # Retrieve artist metadata from Spotify API
        artist = self.sp.artist(artist_id)
        return {
            'artist_id': artist_id,
            'genres': artist['genres'],
            'popularity': artist['popularity']
        }

    def get_track_metadata(self, track_id):
        # Retrieve track metadata from Spotify API
        track = self.sp.track(track_id)
        return {
            'track_id': track_id,
            'title': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'popularity': track['popularity'],
            'duration_ms': track['duration_ms']
        }

    def add_track_features_to_dataset(self, data):
        # Retrieve track metadata and audio features for each track in the dataset
        track_metadata = {}
        track_audio_features = {}
        for _, row in data.iterrows():
            if 'track_id' in row:
                track_id = row['track_id']
                metadata = self.get_track_metadata(track_id)
                audio_features = self.get_track_audio_features(track_id)
                track_metadata[track_id] = metadata
                track_audio_features[track_id] = audio_features

        # Convert track metadata and audio features to DataFrames
        metadata_df = pd.DataFrame.from_dict(track_metadata, orient='index')
        audio_features_df = pd.DataFrame.from_dict(track_audio_features, orient='index')

        # Merge track metadata and audio features with the dataset if they are not empty
        if not metadata_df.empty:
            data = data.merge(metadata_df, on='track_id', how='left')
        if not audio_features_df.empty:
            data = data.merge(audio_features_df, on='track_id', how='left')

        return data