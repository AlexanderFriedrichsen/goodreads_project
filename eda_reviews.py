import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
pd.options.display.float_format = '{:,}'.format


#code samples at bottom of readme:
# https://github.com/MengtingWan/goodreads
# check this file out:
# https://github.com/MengtingWan/goodreads/blob/master/reviews.ipynb

DIR = 'data/'

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
def count_lines(file_name):
    print('counting file:', file_name)
    count = 0
    with gzip.open(file_name) as fin:
        for l in fin:
            count += 1
    print('done!')
    return count

reviews_f = load_data(os.path.join(DIR, 'goodreads_reviews_fantasy_paranormal.json.gz'))
books_f = load_data(os.path.join(DIR, 'goodreads_books_fantasy_paranormal.json.gz'))
interactions_f = load_data(os.path.join(DIR, 'goodreads_interactions_fantasy_paranormal.json.gz'))

reviews = load_data(os.path.join(DIR, 'goodreads_reviews_dedup.json.gz'), None)
books = load_data(os.path.join(DIR, 'goodreads_books.json.gz')) 
# interactions = load_data(os.path.join(DIR, 'goodreads_interactions_fantasy_paranormal.json.gz'))

# works = load_data(os.path.join(DIR, 'goodreads_book_works.json.gz'))
# series = load_data(os.path.join(DIR, 'goodreads_book_series.json.gz'))

# print(' == sample record (fantasy books) ==')
# print(np.random.choice(books_f))
print(' == sample review (fantasy books) ==')
print(reviews_f[0:5])
# print(' == sample interactions (fantasy) ==')
# print(np.random.choice(interactions_f))

#n_book = count_lines(os.path.join(DIR, 'goodreads_books.json.gz'))

#reviews_f_df = pd.DataFrame(reviews_f)
book_reviews_df = pd.DataFrame(reviews)

print(" == loaded review df ==\n\n")

# Calculate the number of words in each review
book_reviews_df['word_count'] = book_reviews_df['review_text'].apply(lambda x: len(x.split()))

# Calculate the number of characters in each review
book_reviews_df['char_count'] = book_reviews_df['review_text'].apply(lambda x: len(x))

# # Convert the 'date_added' column to a datetime object
# book_reviews_df['date_added'] = pd.to_datetime(book_reviews_df['date_added'], format='%a %b %d %H:%M:%S %z %Y')

# # Create a column with the year and month only
# book_reviews_df['year_month'] = book_reviews_df['date_added'].dt.to_period('M')

# Histogram of the number of words in each review
plt.figure(figsize=(10, 6))
sns.histplot(data=book_reviews_df, x='word_count', bins=30)
plt.title('Histogram of the Number of Words in Each Review')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Histogram of the number of characters in each review
plt.figure(figsize=(10, 6))
sns.histplot(data=book_reviews_df, x='char_count', bins=30)
plt.title('Histogram of the Number of Characters in Each Review')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

# Histogram of ratings
plt.figure(figsize=(10, 6))
sns.histplot(data=book_reviews_df, x='rating', discrete=True, bins=5)
plt.title('Histogram of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# # Number of reviews by year and month
# reviews_by_year_month = book_reviews_df['year_month'].value_counts().sort_index()

# plt.figure(figsize=(10, 6))
# reviews_by_year_month.plot(kind='bar')
# plt.title('Number of Reviews by Year and Month')
# plt.xlabel('Year and Month')
# plt.ylabel('Number of Reviews')
# plt.xticks(rotation=45)
# plt.show()


# print(reviews_f_df.columns)
# Total number of reviews
total_reviews = len(book_reviews_df)

# Total number of unique users that left a review
unique_users = book_reviews_df['user_id'].nunique()

# Total number of unique books reviewed
unique_books = book_reviews_df['book_id'].nunique()

print(f"Total number of reviews: {total_reviews}")
print(f"Total number of unique users that left a review: {unique_users}")
print(f"Total number of unique books reviewed: {unique_books}")
