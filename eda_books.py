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

# reviews = load_data(os.path.join(DIR, 'goodreads_reviews_dedup.json.gz'), None)
books = load_data(os.path.join(DIR, 'goodreads_books.json.gz')) #None) 

books_df = pd.DataFrame(books)

print(books_df.columns)
