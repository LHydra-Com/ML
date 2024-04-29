#%%
import pandas as pd
import random

#%%
# Function to get user input
def get_user_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input:
            return user_input
        else:
            print("Please enter a valid input.")

#%%
# Load the dataset
df = pd.read_csv('dataset.csv')

#%%
# Preprocess the data (you can add your custom preprocessing steps here)
df = df.drop_duplicates(subset='usersha1', keep='first')
df['plays'] = (df['plays'] - df['plays'].min()) / (df['plays'].max() - df['plays'].min())  # Normalize 'plays' column

#%%
# Get the artist names and countries
artists = df['artname'].unique().tolist()
countries = df['country'].unique().tolist()

#%%
# Greet the user
print('🤖 Hey there! 😄 Welcome to LHydra, your personal music guru. I am here to help you discover amazing songs that you will absolutely love!')

# Get user input
gender = get_user_input("🤖 So, are you a music man or a melody queen? (Enter 'male' or 'female')\n👤 ")
age = int(get_user_input('🤖 Now, let us talk numbers. How many candles were on your last birthday cake?\n👤 '))
country = get_user_input('🤖 Alright, globetrotter, where are you tuning in from? (Enter your country)\n👤 ')
favorite_artist = get_user_input('🤖 Quick question: who is your all-time favorite artist?\n👤 ')
favorite_song = get_user_input('🤖 And what is that one song by them that you can not stop humming?\n👤 ')
plays = int(get_user_input(f"🤖 Just between us, how many times do you listen to {favorite_artist} on a daily basis?\n👤 "))

#%%
# Print user inputs
print(f"\n🤖 You entered the following information:")
print(f"Gender: {gender}")
print(f"Age: {age}")
print(f"Country: {country}")
print(f"Favorite Artist: {favorite_artist}")
print(f"Favorite Song: {favorite_song}")
print(f"Daily Plays: {plays}")

#%%
# Select a random popular artist from the dataset
popular_artists = df['artname'].value_counts().nlargest(10).index.tolist()
random_artist = random.choice(popular_artists)

# Print the customized message
print(f"\n📚 Did you know that {random_artist} is currently one of the hottest artists on the charts? 🔥")
print(f"📚 With their infectious beats and catchy lyrics, it's no surprise that many people are jamming to their tunes every month! 🎶")

#%%
# Recommend a random song from the dataset
random_song = df.sample(1)['track_name'].values[0]
print(f"\n🤖 Based on your preferences, I recommend checking out '{random_song}' - it's a banger! 🎶")