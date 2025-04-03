"""
Data Preprocessing Script

@Author:
Chong Chen (cstchenc@163.com)

@Created:
25/8/2017

@References:
"""

import os
import json
import pandas as pd
import pickle
import numpy as np

# Set the random seed for reproducibility
np.random.seed(2017)

# Define directories and file paths
# TPS_DIR = '../data/AB'
# TP_file = os.path.join(TPS_DIR, 'All_Beauty_5.json')
# TPS_DIR = '../data/DM'
# TP_file = os.path.join(TPS_DIR, 'Digital_Music_5.json')
TPS_DIR = '../data/TG'
TP_file = os.path.join(TPS_DIR, 'Toys_and_Games_5.json')
# TPS_DIR = '../data/music'
# TP_file = os.path.join(TPS_DIR, 'Digital_Music_5.json')

# Initialize lists to store data
users_id = []
items_id = []
ratings = []
reviews = []

# Function to clean IDs by stripping whitespace and trailing commas
def clean_id(id_str):
    return id_str.strip().rstrip(',')

# Read and process the JSON file
with open(TP_file, 'r', encoding='utf-8') as f:
    for line in f:
        js = json.loads(line)
        reviewer_id = str(js.get('reviewerID', 'unknown')).strip()
        asin = str(js.get('asin', 'unknown')).strip()
        
        # Skip entries with 'unknown' reviewerID or asin
        if reviewer_id.lower() == 'unknown':
            print("Skipping entry with unknown reviewerID.")
            continue
        if asin.lower() == 'unknown':
            print("Skipping entry with unknown asin.")
            continue
        
        # Append cleaned data without trailing commas
        reviews.append(js.get('reviewText', ''))
        users_id.append(clean_id(reviewer_id))
        items_id.append(clean_id(asin))
        ratings.append(js.get('overall', 0))  # Assuming 'overall' is numerical

# Create the DataFrame
data = pd.DataFrame({
    'user_id': pd.Series(users_id),
    'item_id': pd.Series(items_id),
    'ratings': pd.Series(ratings),
    'reviews': pd.Series(reviews)
})[['user_id', 'item_id', 'ratings', 'reviews']]

# Function to count occurrences
def get_count(tp, id_column):
    count = tp.groupby(id_column).size().reset_index(name='count')
    return count

# Get user and item counts
usercount = get_count(data, 'user_id')
itemcount = get_count(data, 'item_id')

# Extract unique user and item IDs
unique_uid = usercount['user_id'].tolist()
unique_sid = itemcount['item_id'].tolist()

# Create mapping dictionaries without trailing commas
user2id = {uid: i for i, uid in enumerate(unique_uid)}
item2id = {sid: i for i, sid in enumerate(unique_sid)}

# Function to numerize user_id and item_id
def numerize(tp):
    # Map user_id and item_id to numerical IDs using the mapping dictionaries
    tp['user_id'] = tp['user_id'].map(user2id).fillna(-1).astype(int)
    tp['item_id'] = tp['item_id'].map(item2id).fillna(-1).astype(int)
    return tp

# Apply the numerize function
data = numerize(data)

# Extract ratings for splitting
tp_rating = data[['user_id', 'item_id', 'ratings']]

# Split the data into train and temporary (test+validation) sets
n_ratings = tp_rating.shape[0]
test_size = int(0.20 * n_ratings)
test_indices = np.random.choice(n_ratings, size=test_size, replace=False)
test_mask = np.zeros(n_ratings, dtype=bool)
test_mask[test_indices] = True

tp_train = tp_rating[~test_mask]
tp_temp = tp_rating[test_mask]
data_train = data[~test_mask]
data_temp = data[test_mask]

# Further split the temporary set into test and validation sets
n_temp = tp_temp.shape[0]
valid_size = int(0.50 * n_temp)
valid_indices = np.random.choice(n_temp, size=valid_size, replace=False)
valid_mask = np.zeros(n_temp, dtype=bool)
valid_mask[valid_indices] = True

tp_valid = tp_temp[valid_mask]
tp_test = tp_temp[~valid_mask]
data_valid = data_temp[valid_mask]
data_test = data_temp[~valid_mask]

# Save the splits to CSV files without headers and indices
tp_train.to_csv(os.path.join(TPS_DIR, 'train.csv'), index=False, header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'valid.csv'), index=False, header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'test.csv'), index=False, header=None)

# Initialize dictionaries to store reviews and related IDs
user_reviews = {}
item_reviews = {}
user_rid = {}
item_rid = {}

# Populate the dictionaries with training data
for row in data_train.itertuples(index=False):
    user_id, item_id, rating, review = row
    # User reviews and item reviews
    user_reviews.setdefault(user_id, []).append(review)
    user_rid.setdefault(user_id, []).append(item_id)
    item_reviews.setdefault(item_id, []).append(review)
    item_rid.setdefault(item_id, []).append(user_id)

# Handle the temporary (test+validation) data to ensure all users and items exist in the dictionaries
for row in data_temp.itertuples(index=False):
    user_id, item_id, rating, review = row
    # Check and initialize missing users
    if user_id not in user_reviews:
        user_reviews[user_id] = ['0']
        user_rid[user_id] = [0]
    # Check and initialize missing items
    if item_id not in item_reviews:
        item_reviews[item_id] = ['0']
        item_rid[item_id] = [0]

# Save the dictionaries using pickle
with open(os.path.join(TPS_DIR, 'user_review'), 'wb') as f:
    pickle.dump(user_reviews, f)

with open(os.path.join(TPS_DIR, 'item_review'), 'wb') as f:
    pickle.dump(item_reviews, f)

with open(os.path.join(TPS_DIR, 'user_rid'), 'wb') as f:
    pickle.dump(user_rid, f)

with open(os.path.join(TPS_DIR, 'item_rid'), 'wb') as f:
    pickle.dump(item_rid, f)

# Recalculate counts after splitting
usercount_final = get_count(data_train, 'user_id')
itemcount_final = get_count(data_train, 'item_id')

# Display sorted counts
print("User Counts:", np.sort(usercount_final['count'].values))
print("Item Counts:", np.sort(itemcount_final['count'].values))
