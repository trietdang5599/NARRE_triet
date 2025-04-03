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
import re
import itertools
from collections import Counter
import argparse

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    u_text_padded = {}
    for user_id, reviews in u_text.items():
        padded_reviews = []
        for i in range(u_len):
            if i < len(reviews):
                review = reviews[i]
                if len(review) < u2_len:
                    # Pad the review
                    padded_review = review + [padding_word] * (u2_len - len(review))
                else:
                    # Truncate the review
                    padded_review = review[:u2_len]
                padded_reviews.append(padded_review)
            else:
                # Pad with empty reviews
                padded_reviews.append([padding_word] * u2_len)
        u_text_padded[user_id] = padded_reviews
    return u_text_padded


def pad_reviewid(u_train, u_valid, u_len, num):
    """
    Pads review IDs to a fixed length.
    """
    def pad_ids(id_list):
        padded_ids = []
        for ids in id_list:
            if len(ids) < u_len:
                # Pad with the specified number
                padded = ids + [num] * (u_len - len(ids))
            else:
                # Truncate the list
                padded = ids[:u_len]
            padded_ids.append(padded)
        return padded_ids

    pad_u_train = pad_ids(u_train)
    pad_u_valid = pad_ids(u_valid)
    return pad_u_train, pad_u_valid


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary for users
    word_counts1 = Counter(itertools.chain(*sentences1))
    vocabulary_inv1 = sorted(word_counts1.keys())
    vocabulary1 = {word: idx for idx, word in enumerate(vocabulary_inv1)}

    # Build vocabulary for items
    word_counts2 = Counter(itertools.chain(*sentences2))
    vocabulary_inv2 = sorted(word_counts2.keys())
    vocabulary2 = {word: idx for idx, word in enumerate(vocabulary_inv2)}

    return vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentences to indices based on the vocabulary.
    """
    u_text_mapped = {}
    for user_id, reviews in u_text.items():
        u_text_mapped[user_id] = np.array([[vocabulary_u.get(word, 0) for word in review] for review in reviews])

    i_text_mapped = {}
    for item_id, reviews in i_text.items():
        i_text_mapped[item_id] = np.array([[vocabulary_i.get(word, 0) for word in review] for review in reviews])

    return u_text_mapped, i_text_mapped


def load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords):
    """
    Loads data from files and processes it.
    """
    # Load pickled review data
    with open(user_review, "rb") as f1, \
         open(item_review, "rb") as f2, \
         open(user_rid, "rb") as f3, \
         open(item_rid, "rb") as f4:
        user_reviews = pickle.load(f1)
        item_reviews = pickle.load(f2)
        user_rids = pickle.load(f3)
        item_rids = pickle.load(f4)

    # Initialize dictionaries and lists
    reid_user_train = []
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []
    u_text = {}
    u_rid = {}
    i_text = {}
    i_rid = {}

    # Process training data
    with open(train_data, "r", encoding='utf-8') as f_train:
        for line_num, line in enumerate(f_train, 1):
            line = line.strip().split(',')
            if len(line) < 3:
                print(f"Skipping malformed line {line_num} in training data.")
                continue

            uid = int(line[0])
            iid = int(line[1])
            rating = float(line[2])

            uid_train.append(uid)
            iid_train.append(iid)
            y_train.append(rating)

            # Process user reviews
            if uid in u_text:
                reid_user_train.append(u_rid[uid])
            else:
                # Clean and tokenize user reviews
                u_text[uid] = [clean_str(s).split() for s in user_reviews.get(uid, ['<PAD/>'])]
                # Map user review IDs, default to 0 if not present
                u_rid[uid] = [int(s) for s in user_rids.get(uid, [0])]
                reid_user_train.append(u_rid[uid])

            # Process item reviews
            if iid in i_text:
                reid_item_train.append(i_rid[iid])
            else:
                # Clean and tokenize item reviews
                i_text[iid] = [clean_str(s).split() for s in item_reviews.get(iid, ['<PAD/>'])]
                # Map item review IDs, default to 0 if not present
                i_rid[iid] = [int(s) for s in item_rids.get(iid, [0])]
                reid_item_train.append(i_rid[iid])

    print("Processing validation data")

    # Initialize validation data lists
    reid_user_valid = []
    reid_item_valid = []
    uid_valid = []
    iid_valid = []
    y_valid = []

    # Process validation data
    with open(valid_data, "r", encoding='utf-8') as f_valid:
        for line_num, line in enumerate(f_valid, 1):
            line = line.strip().split(',')
            if len(line) < 3:
                print(f"Skipping malformed line {line_num} in validation data.")
                continue

            uid = int(line[0])
            iid = int(line[1])
            rating = float(line[2])

            uid_valid.append(uid)
            iid_valid.append(iid)
            y_valid.append(rating)

            # Handle users in validation data
            if uid in u_text:
                reid_user_valid.append(u_rid[uid])
            else:
                # Initialize with padding if user not in training data
                u_text[uid] = [['<PAD/>']]
                u_rid[uid] = [0]
                reid_user_valid.append(u_rid[uid])

            # Handle items in validation data
            if iid in i_text:
                reid_item_valid.append(i_rid[iid])
            else:
                # Initialize with padding if item not in training data
                i_text[iid] = [['<PAD/>']]
                i_rid[iid] = [0]
                reid_item_valid.append(i_rid[iid])

    # Determine padding lengths based on the 90th percentile
    review_num_u = np.array([len(reviews) for reviews in u_text.values()])
    u_len = int(np.percentile(review_num_u, 90))
    review_len_u = np.array([len(review) for reviews in u_text.values() for review in reviews])
    u2_len = int(np.percentile(review_len_u, 90))

    review_num_i = np.array([len(reviews) for reviews in i_text.values()])
    i_len = int(np.percentile(review_num_i, 90))
    review_len_i = np.array([len(review) for reviews in i_text.values() for review in reviews])
    i2_len = int(np.percentile(review_len_i, 90))

    print(f"u_len: {u_len}")
    print(f"i_len: {i_len}")
    print(f"u2_len: {u2_len}")
    print(f"i2_len: {i2_len}")

    user_num = len(u_text)
    item_num = len(i_text)
    print(f"user_num: {user_num}")
    print(f"item_num: {item_num}")

    return [u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, 
            uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, 
            reid_user_train, reid_item_train, reid_user_valid, reid_item_valid]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    u_text_mapped = {}
    for user_id, reviews in u_text.items():
        u_text_mapped[user_id] = np.array([[vocabulary_u.get(word, 0) for word in review] for review in reviews])

    i_text_mapped = {}
    for item_id, reviews in i_text.items():
        i_text_mapped[item_id] = np.array([[vocabulary_i.get(word, 0) for word in review] for review in reviews])

    return u_text_mapped, i_text_mapped


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary for users
    word_counts1 = Counter(itertools.chain(*sentences1))
    vocabulary_inv1 = sorted(word_counts1.keys())
    vocabulary1 = {word: idx for idx, word in enumerate(vocabulary_inv1)}

    # Build vocabulary for items
    word_counts2 = Counter(itertools.chain(*sentences2))
    vocabulary_inv2 = sorted(word_counts2.keys())
    vocabulary2 = {word: idx for idx, word in enumerate(vocabulary_inv2)}

    return vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    u_text_mapped = {}
    for user_id, reviews in u_text.items():
        u_text_mapped[user_id] = np.array([
            [vocabulary_u.get(word, 0) for word in review]
            for review in reviews
        ])
    
    i_text_mapped = {}
    for item_id, reviews in i_text.items():
        i_text_mapped[item_id] = np.array([
            [vocabulary_i.get(word, 0) for word in review]
            for review in reviews
        ])

    return u_text_mapped, i_text_mapped



def load_data(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords):
    """
    Loads and preprocesses data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    data = load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, stopwords)
    (u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, 
     uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, 
     reid_user_train, reid_item_train, reid_user_valid, reid_item_valid) = data

    print("Load data done")

    # Pad sentences and review IDs
    u_text = pad_sentences(u_text, u_len, u2_len)
    reid_user_train, reid_user_valid = pad_reviewid(reid_user_train, reid_user_valid, u_len, item_num + 1)

    print("Pad user done")

    i_text = pad_sentences(i_text, i_len, i2_len)
    reid_item_train, reid_item_valid = pad_reviewid(reid_item_train, reid_item_valid, i_len, user_num + 1)

    print("Pad item done")

    # Build vocabularies
    user_voc = [word for reviews in u_text.values() for review in reviews for word in review]
    item_voc = [word for reviews in i_text.values() for review in reviews for word in review]

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print(f"User Vocabulary Size: {len(vocabulary_user)}")
    print(f"Item Vocabulary Size: {len(vocabulary_item)}")

    # Map words to indices
    u_text_mapped, i_text_mapped = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)

    # Convert lists to numpy arrays
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    iid_train = np.array(iid_train)
    uid_valid = np.array(uid_valid)
    iid_valid = np.array(iid_valid)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)

    return [u_text_mapped, i_text_mapped, y_train, y_valid, vocabulary_user, 
            vocabulary_inv_user, vocabulary_item, vocabulary_inv_item, 
            uid_train, iid_train, uid_valid, iid_valid, user_num, 
            item_num, reid_user_train, reid_item_train, 
            reid_user_valid, reid_item_valid]


def main(args):
    # TPS_DIR = '../data/AB'
    # TPS_DIR = '../data/DM'
    TPS_DIR = '../data/TG'

    u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, \
    vocabulary_item, vocabulary_inv_item, uid_train, iid_train, uid_valid, \
    iid_valid, user_num, item_num, reid_user_train, reid_item_train, \
    reid_user_valid, reid_item_valid = load_data(
        args.train_data,
        args.valid_data,
        args.user_review,
        args.item_review,
        args.user_review_id,
        args.item_review_id,
        args.stopwords
    )
    print(u_text[0])
    np.random.seed(2017)

    # Shuffle the training data
    shuffle_indices = np.random.permutation(len(y_train))
    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train_shuffled = y_train[shuffle_indices]
    reid_user_train_shuffled = reid_user_train[shuffle_indices]
    reid_item_train_shuffled = reid_item_train[shuffle_indices]

    # Reshape for consistency
    y_train_shuffled = y_train_shuffled[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    # Create batches as tuples
    batches_train = list(zip(userid_train, itemid_train, reid_user_train_shuffled, reid_item_train_shuffled, y_train_shuffled))
    batches_test = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))

    print('Write begin')

    # Save the batches using pickle
    with open(os.path.join(TPS_DIR, 'dataset.train'), 'wb') as output:
        pickle.dump(batches_train, output)

    with open(os.path.join(TPS_DIR, 'dataset.test'), 'wb') as output:
        pickle.dump(batches_test, output)

    # Prepare parameters to save
    para = {
        'user_num': user_num,
        'item_num': item_num,
        'review_num_u': u_text[0].shape[0],
        'review_num_i': i_text[0].shape[0],
        'review_len_u': u_text[1].shape[1],
        'review_len_i': i_text[1].shape[1],
        'user_vocab': vocabulary_user,
        'item_vocab': vocabulary_item,
        'train_length': len(y_train),
        'test_length': len(y_valid),
        'u_text': u_text,
        'i_text': i_text
    }

    with open(os.path.join(TPS_DIR, 'dataset.para'), 'wb') as output:
        # Pickle dictionary using the highest protocol for efficiency
        pickle.dump(para, output, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data preprocessing completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing for Music Recommender')
    # parser.add_argument("--valid_data", type=str, default="../data/AB/valid.csv", help="Data for validation")
    # parser.add_argument("--test_data", type=str, default="../data/AB/test.csv", help="Data for testing")
    # parser.add_argument("--train_data", type=str, default="../data/AB/train.csv", help="Data for training")
    # parser.add_argument("--user_review", type=str, default="../data/AB/user_review", help="User's reviews")
    # parser.add_argument("--item_review", type=str, default="../data/AB/item_review", help="Item's reviews")
    # parser.add_argument("--user_review_id", type=str, default="../data/AB/user_rid", help="User review IDs")
    # parser.add_argument("--item_review_id", type=str, default="../data/AB/item_rid", help="Item review IDs")
    
    # parser.add_argument("--valid_data", type=str, default="../data/DM/valid.csv", help="Data for validation")
    # parser.add_argument("--test_data", type=str, default="../data/DM/test.csv", help="Data for testing")
    # parser.add_argument("--train_data", type=str, default="../data/DM/train.csv", help="Data for training")
    # parser.add_argument("--user_review", type=str, default="../data/DM/user_review", help="User's reviews")
    # parser.add_argument("--item_review", type=str, default="../data/DM/item_review", help="Item's reviews")
    # parser.add_argument("--user_review_id", type=str, default="../data/DM/user_rid", help="User review IDs")
    # parser.add_argument("--item_review_id", type=str, default="../data/DM/item_rid", help="Item review IDs")
    
    parser.add_argument("--valid_data", type=str, default="../data/TG/valid.csv", help="Data for validation")
    parser.add_argument("--test_data", type=str, default="../data/TG/test.csv", help="Data for testing")
    parser.add_argument("--train_data", type=str, default="../data/TG/train.csv", help="Data for training")
    parser.add_argument("--user_review", type=str, default="../data/TG/user_review", help="User's reviews")
    parser.add_argument("--item_review", type=str, default="../data/TG/item_review", help="Item's reviews")
    parser.add_argument("--user_review_id", type=str, default="../data/TG/user_rid", help="User review IDs")
    parser.add_argument("--item_review_id", type=str, default="../data/TG/item_rid", help="Item review IDs")
    parser.add_argument("--stopwords", type=str, default="../data/stopwords", help="Stopwords file")
    print("Data preprocessing started")
    args = parser.parse_args()

    main(args)
