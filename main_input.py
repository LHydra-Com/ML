#%%
# import libraries
from model import HybridRecommender
from music_data import Music
from ml.data_utils import preprocess_data, extract_user_item_interactions, extract_item_content_features, split_data
import pandas as pd
import time
from statistics import mode
import warnings
import pycountry
import ast
# Ignore all warnings
warnings.filterwarnings("ignore")
#%%
# Extract the various artists in the dataset and their associated encoded codes
start_time = time.time()
df = pd.read_csv('dataset.csv')
df = df.drop_duplicates(subset='usersha1', keep='first')

#%%
# Normalize the 'plays' column
min_value = df['plays'].min()
max_value = df['plays'].max()
df['plays'] = (df['plays'] - min_value) / (max_value - min_value)
df.to_csv('normalized_data.csv')

#%%
# Get the artist names and countries
artists = df['artname'].unique().tolist()
artists_dict = {artist: i for i, artist in enumerate(artists)}
countries = df['country'].unique().tolist()
countries_dict = {country: i for i, country in enumerate(countries)}

#%%
# Function to get an artist name given the artist encoding code
def artname_by_artcode(artist_code, dictionary=artists_dict):
    for key, value in dictionary.items():
        if value == artist_code:
            return key
    msg = f'No artist found with code {artist_code}.'
    return msg

#%%
end_time = time.time()      
execution_time = end_time - start_time

if execution_time < 20:
    print(f"🤖 Whew, that was a swift {int(execution_time)} seconds! Time flies when you're grooving to the beat! 🕰️🎵")
else:
    minutes, seconds = divmod(int(execution_time), 20)
    print(f"🤖 Wow, we've been jamming for {minutes} minutes and {seconds} seconds already! Music makes time fly, doesn't it? 🕰️🎶")
print('🤖 Hey there! 😄 Welcome to LHydra, your personal music guru. I am here to help you discover amazing songs that you will absolutely love!')
print("🤖 But first, let me get to know you a little better. Don't worry, I won't bite! 😉")

#%%
# Get user input from a list
user_data = ["male", 25, "United States", "Led Zeppelin", "Stairway to Heaven", 5]

gender, age, country, favorite_artist, favorite_song, plays = user_data

if gender.lower() == 'male':
    gender = 1
    print('🤖 Awesome! A music maestro in the house! 🎵')
elif gender.lower() == 'female':
    gender = 2
    print('🤖 Fantastic! A songstress extraordinaire! 🎶')
else:
    print('🤖 Oops!, {gender} is not a valid input. Let us try again and stick to male or female, shall we? 😅')

print(f"🤖 {age} years? That's a great age to explore some fresh tunes! 🎉")

entered_country = country.title()

try:
    country_obj = pycountry.countries.get(name=entered_country)
    capital = country_obj.capital
    country_code = country_obj.alpha_3
    country = countries_dict[country_code]
    print(f"🤖 Wow, {entered_country}! The capital {capital} is known for its vibrant music culture! 🌍")
    print(f"🤖 I hear the music scene in {entered_country} is incredible! 🎵")
except KeyError:
    print(f"🤖 Hmm, I couldn't find information for {entered_country}. Please try entering the official country name or check the spelling.")
    country = None
except:
    print("🤖 Oops, something went wrong while fetching country information. Let's move on.")
    country = None

print(f"🤖 No way, {favorite_artist} is legendary! And {favorite_song} - is an absolute gem! 💎")
#%%
monthly_plays = plays * 30

average_playcounts = round(df['plays'].mean())
global_popular_artist = df['artname'].mode()[0]
print(f"🤖 Wow, you listen to your {favorite_song} an average of {plays} times a day? That's impressive! 🎧")
if monthly_plays > 1000:
    print(f"🤖 Wow, you'll be grooving to your favorite artist's tunes around {monthly_plays:,} times a month! 🎉 That's some serious dedication! 🎧")
else:
    print(f"🤖 If you keep that up, you'll be enjoying your favorite artist's music around {monthly_plays} times a month! 🎶 Keep vibing! 😄")

print(f"🤖 Did you know that {global_popular_artist} is currently the hottest artist on the charts? 🔥")
print(f"🤖 With their infectious beats and catchy lyrics, it's no surprise that many people are jamming to their tunes every month! 🎶")

print(f"🤖 Ok! Enough of my fun facts! Here comes my findings")

#%%
# Impute artist, if not provided
if favorite_artist.lower() in ['none', 'unknown']:
    fav_artist = global_popular_artist
    fav_artist = artists_dict[fav_artist]
elif favorite_artist in artists_dict:
    fav_artist = artists_dict[favorite_artist]
else:
    new_artist_number = len(artists_dict) + 1
    artists_dict[favorite_artist] = new_artist_number
    fav_artist = artists_dict[favorite_artist]

# Impute play counts with global average, if not provided
if plays in [0, 'none', 'None']:
    plays = average_playcounts
else:
    plays = int(plays)
#%%
    
# Load and preprocess the dataset
data = pd.read_csv('normalized_data.csv')
music = Music()
data = music.add_track_features_to_dataset(data)
preprocessed_data = preprocess_data(data)

# Extract user-item interactions and item content features
user_item_interactions = extract_user_item_interactions(preprocessed_data)
item_content_features = extract_item_content_features(preprocessed_data)

# Split the data into training and testing sets using cross-validation
train_data_list, test_data_list = split_data(user_item_interactions, n_splits=5, random_state=42)

# Create an instance of the HybridRecommender
recommender = HybridRecommender(user_item_interactions, item_content_features)

#%%
# Perform cross-validation and evaluate the recommender system
evaluation_metrics = []
for train_data, test_data in zip(train_data_list, test_data_list):
    # Train the recommender system using the training data
    recommender.train(train_data)
    
    # Evaluate the recommender system using the testing data
    metrics = recommender.evaluate(test_data)
    evaluation_metrics.append(metrics)
#%%
# Calculate average evaluation metrics across all cross-validation splits
avg_metrics = {metric: sum(values) / len(values) for metric, values in zip(evaluation_metrics[0].keys(), zip(*[d.values() for d in evaluation_metrics]))}

print("Average Evaluation Metrics (Cross-Validation):")
print(f"NDCG: {avg_metrics['NDCG']}")
print(f"MRR: {avg_metrics['MRR']}")
print(f"Precision: {avg_metrics['Precision']}")
print(f"Recall: {avg_metrics['Recall']}")
print(f"F1-score: {avg_metrics['F1-score']}")
print(f"MAP: {avg_metrics['MAP']}")

#%%
# Retrieve user listening history
user_id = "user123"  # Example user ID
listening_history = music.get_user_listening_history(user_id)
#%%
# Retrieve track audio features
track_name = "Stairway to Heaven"  # Example track name
track_results = music.sp.search(q=track_name, type='track', limit=1)
if track_results['tracks']['items']:
    track_id = track_results['tracks']['items'][0]['id']
    audio_features = music.get_track_audio_features(track_id)
else:
    print("Track not found.")
#%%
    
# Retrieve artist metadata
artist_name = "Led Zeppelin"  # Example artist name
artist_results = music.sp.search(q=artist_name, type='artist', limit=1)
if artist_results['artists']['items']:
    artist_id = artist_results['artists']['items'][0]['id']
    artist_metadata = music.get_artist_metadata(artist_id)
else:
    print("Artist not found.")
#%%
# Retrieve track metadata
track_metadata = music.get_track_metadata(track_id)

# Generate recommendations using the hybrid approach
recommendation_start_time = time.time()
recommendations = recommender.generate_recommendations(user_id, top_n=10)

# After generating recommendations
recommendation_end_time = time.time()
recommendation_time = recommendation_end_time - recommendation_start_time

if recommendations:
    if recommendation_time < 60:
        print(f"Drumroll please... 🥁 Let's see what LHydra has discovered for you in just {int(recommendation_time)} seconds! 🎉")
    else:
        minutes, seconds = divmod(int(recommendation_time), 60)
        print(f"Drumroll please... 🥁 After {minutes} minutes and {seconds} seconds of intense calculation, LHydra has some stellar recommendations for you! 🎉")
    print('=' * 40)
    for item in recommendations:
        print(f"- {item}")
else:
    print("🤖 Oops! 🙈 It looks like LHydra couldn't find any personalized recommendations for you at the moment.")
    print("🤖 But don't worry, I've got a backup plan! Check out these trending items that are sure to make your playlist pop! 🎉")
    trending_items = recommender.get_trending_items()
    for item in trending_items:
        print(f"- {item}")

print("🤖 Remember, music is a journey, and there's always something new to discover! 🌟")
print("🤖 Keep exploring, and don't be afraid to step out of your comfort zone! 🎧")
print("🤖 Until next time, happy listening! 😄🎶")