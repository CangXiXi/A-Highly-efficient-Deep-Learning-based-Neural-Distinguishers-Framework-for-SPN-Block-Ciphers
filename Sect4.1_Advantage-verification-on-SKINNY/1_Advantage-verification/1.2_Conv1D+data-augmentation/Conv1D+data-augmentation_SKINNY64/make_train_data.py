from os import urandom
import numpy as np
from skinny import Skinny

def convert_to_binary(arr, n_words, word_size):
    sample_len = 3 * n_words * word_size
    n_samples = len(arr[0])
    x = np.zeros((sample_len, n_samples), dtype=np.uint8)
    for i in range(sample_len):
        index = i // word_size
        offset = word_size - (i % word_size) - 1
        x[i] = (arr[index] >> offset) & 1
    x = x.transpose()
    return x

def make_train_data(n_samples, cipher, diff, case=0, y=None, additional_conditions=None):
    if y is None:
        y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    elif y == 0 or y == 1:
        y = np.array([y for _ in range(n_samples)])
    # draw keys and plaintexts
    keys = cipher.draw_keys(n_samples)
    pt0 = cipher.draw_plaintexts(n_samples)
    if additional_conditions is not None:
        pt0 = additional_conditions(pt0)
    pt1 = pt0 ^ np.array(diff, dtype=cipher.word_dtype)[:, np.newaxis]
    num_rand_samples = np.sum(y == 0)
    pt1[:, y == 0] = cipher.draw_plaintexts(num_rand_samples)
    ct0 = cipher.encrypt(pt0, keys)
    ct1 = cipher.encrypt(pt1, keys)
    if case != 0:
        ct0 = cipher.data_aug(ct0, pt0, case)
        ct1 = cipher.data_aug(ct1, pt1, case)
    x = convert_to_binary(np.concatenate((ct0, ct1,(np.array(ct0) ^ np.array(ct1))), axis=0), cipher.get_n_words(), cipher.get_word_size())
    return x, y

def make_mult_pairs_data(n_samples, cipher, diff, case=0, n_pairs=1):
    y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    y_atomic = np.repeat(y, n_pairs)
    keys = cipher.draw_keys(n_samples * n_pairs)
    pt0 = cipher.draw_plaintexts(n_samples * n_pairs)
    pt1 = pt0 ^ np.array(diff, dtype=cipher.word_dtype)[:, np.newaxis]
    num_rand_samples = np.sum(y_atomic == 0)
    pt1[:, y_atomic == 0] = cipher.draw_plaintexts(num_rand_samples)
    ct0 = cipher.encrypt(pt0, keys)
    ct1 = cipher.encrypt(pt1, keys)
    if case != 0:
        ct0 = cipher.data_aug(ct0, pt0, case)
        ct1 = cipher.data_aug(ct1, pt1, case)
    x = convert_to_binary(np.concatenate((ct0, ct1,(np.array(ct0) ^ np.array(ct1))), axis=0), cipher.get_n_words(), cipher.get_word_size(),n_pairs)
    x = x.reshape((-1, n_pairs, n_samples))
    return x, y