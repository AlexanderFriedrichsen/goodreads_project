import gzip
import json
import re
import os
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
from textblob import TextBlob
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
pd.options.display.float_format = '{:,}'.format

# download the NLTK stopwords and wordnet corpora
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("downloaded package wordnet")

DIR = 'data/'

# functin to load in the goodreads gzip files
def load_data(file_name, head = 500):
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            data.append(d)
            
            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    return data

reviews = load_data(os.path.join(DIR, 'goodreads_reviews_dedup.json.gz'), 1000)
reviews_df = pd.DataFrame(reviews)

# new column for full review sentiment polarity score
reviews_df['sentiment_polarity'] = 0.0
# new column for sentence-wise sentiment polarity score
reviews_df['sentiment_polarity_sentence'] = ''


# create a new DataFrame with a random subset of 1000 reviews
# we need to perform analyses with speed for bugfixing, before doing the full dataset.
num_reviews = len(reviews_df)
subset_size = 1000
if num_reviews > subset_size:
    random_indices = random.sample(range(num_reviews), subset_size)
    reviews_subset_df = reviews_df.iloc[random_indices, :]
else:
    reviews_subset_df = reviews_df.copy() # if there are less than 1000 reviews, use all of them

# perform preprocessing on a review text string
def preprocess_reviews(reviews):
    preprocessed_reviews = []
    for review in tqdm(reviews, desc='Preprocessing Reviews', unit='review'):
        # convert to lowercase
        review = review.lower()
        # remove punctuation
        review = review.translate(str.maketrans('', '', string.punctuation))
        # tokenize into words
        words = word_tokenize(review)
        # remove stop words
        words = [word for word in words if word not in stopwords.words('english')]
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        # join the words back into a string
        review = ' '.join(words)
        preprocessed_reviews.append(review)
    return preprocessed_reviews

# preprocess the subset of reviews using the preprocess_reviews function
preprocessed_reviews = preprocess_reviews(reviews_subset_df)

# create a DataFrame to store the preprocessed reviews
preprocessed_reviews_df = pd.DataFrame({'review_id': reviews_subset_df.index, 'review_text_preprocessed': preprocessed_reviews})

# merge the preprocessed reviews DataFrame back into the original reviews DataFrame
reviews_df = pd.merge(reviews_df, preprocessed_reviews_df, on='review_id')


# preprocess the review_text column of the DataFrame
reviews_subset_df['review_text_preprocessed'] = reviews_subset_df['review_text'].apply(preprocess_review)
print("reviews text preprocessed")

# calculate the sentiment polarity of each review using TextBlob
reviews_subset_df['sentiment_polarity'] = reviews_subset_df['review_text_preprocessed'].apply(lambda x: TextBlob(x).sentiment.polarity)

# calculate the absolute value of the sentiment polarity for each review
reviews_subset_df['sentiment_polarity_abs'] = reviews_subset_df['sentiment_polarity'].abs()

# def controversial_detection(reviews_df):
    # # calculate the % cutoff for extremely positive and negative reviews
    # num_reviews = len(reviews_subset_df)
    # cutoff_pos = reviews_subset_df['sentiment_polarity_abs'].quantile(0.80)
    # cutoff_neg = reviews_subset_df['sentiment_polarity_abs'].quantile(0.2)

    # # group the reviews by book_id and calculate the mean sentiment polarity and the count of extremely positive and negative reviews
    # grouped_df = reviews_subset_df.groupby('book_id').agg({'sentiment_polarity': 'mean', 'sentiment_polarity_abs': 'count'})

    # # filter the grouped DataFrame to only include books with at least one extremely positive and one extremely negative review
    # controversial_books = grouped_df[(grouped_df['sentiment_polarity'] >= cutoff_pos) & (grouped_df['sentiment_polarity'] <= cutoff_neg)]
    # # print the names of the controversial books
    # print(controversial_books.index.tolist())

    # # select the first controversial book and print its reviews
    # book_id = controversial_books.index[0]
    # book_reviews = reviews_subset_df[reviews_subset_df['book_id'] == book_id]['review_text']
    # for review in book_reviews:
    #     print(review)

def plot_sentiment_polarity(reviews_df):            
    # plot a histogram of the sentiment polarity distribution
    plt.hist(reviews_df['sentiment_polarity'], bins=50)
    plt.title('Sentiment Polarity Distribution')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Count')
    plt.show()

plot_sentiment_polarity(reviews_subset_df)
