import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load the dataset
data = pd.read_excel('Netflx Review.xlsx')

# Clean text
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower()
    return text

# Tokenize text
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# Remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Apply preprocessing steps to the dataset
data['Review_Text'] = data['Review_Text'].apply(clean_text)
data['tokens'] = data['Review_Text'].apply(tokenize_text)
data['tokens'] = data['tokens'].apply(remove_stopwords)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to determine sentiment
def analyze_sentiment(text):
    sentiment = sid.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'positive'
    elif sentiment['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis to the 'review_text' column
data['sentiment'] = data['Review_Text'].apply(analyze_sentiment)

data.to_excel('preprocessed_data.xlsx', index=False)
