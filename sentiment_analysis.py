import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

data = pd.read_excel('preprocessed_data.xlsx')

#Intialize Sentiment Analysis
sid = SentimentIntensityAnalyzer()

#Function to determine sentiment
def analyze_sentiment(text):
    sentiment = sid.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'positive'
    elif sentiment['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'
    
# Apply sentiment analysis to the 'review_text' column and update 'Sentiment' column
data['Sentiment'] = data['Review_Text'].apply(analyze_sentiment)

# Update the 'sentiment' column to match the newly assigned labels
data['sentiment'] = data['Sentiment']

# Save the data with sentiment labels
data.to_excel('sentiment_analyzed_data.xlsx', index=False)
