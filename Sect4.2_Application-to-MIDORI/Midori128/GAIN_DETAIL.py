from abc import ABC, abstractmethod
import numpy as np
from os import urandom



class AbstractCipher(ABC):
    """ Abstract cipher class containing all methods a cipher class should implement """

    """ Data types for all the supported word sizes """
    DTYPES = {
        2: np.uint8,
        4: np.uint8,
        8: np.uint8,
        16: np.uint16,
        32: np.uint32
    }

    def __init__(
            self, n_rounds, word_size, n_words, n_main_key_words, n_round_key_words,
            use_key_schedule=True, main_key_word_size=None, round_key_word_size=None
    ):
        self.n_rounds = n_rounds
        self.word_size = word_size
        self.word_dtype = self.DTYPES.get(self.word_size, None)
        if self.word_dtype is None:
            raise Exception(f'Error: Unexpected word size {self.word_size}')
        self.mask_val = 2 ** self.word_size - 1
        self.n_words = n_words
        self.n_main_key_words = n_main_key_words
        self.n_round_key_words = n_round_key_words
        self.use_key_schedule = use_key_schedule
        self.main_key_word_size = main_key_word_size if main_key_word_size is not None else word_size
        self.main_key_word_dtype = self.DTYPES.get(self.main_key_word_size, None)
        if self.main_key_word_dtype is None:
            raise Exception(f'Error: Unexpected word size {self.main_key_word_size}')
        self.round_key_word_size = round_key_word_size if round_key_word_size is not None else word_size
        self.round_key_word_dtype = self.DTYPES.get(self.round_key_word_size, None)
        if self.round_key_word_dtype is None:
            raise Exception(f'Error: Unexpected word size {self.round_key_word_size}')

    def get_word_size(self):
        return self.word_size

    def get_n_words(self):
        return self.n_words

    def get_block_size(self):
        return self.word_size * self.n_words

    def get_n_rounds(self):
        return self.n_rounds

    def set_n_rounds(self, new_n_rounds):
        self.n_rounds = new_n_rounds

    @staticmethod
    def bytes_per_word(word_size):
        return word_size // 8 + (1 if (word_size % 8) else 0)

    @abstractmethod
    def encrypt_one_round(self, p, k, rc=None):
        pass

    def encrypt(self, p, keys):
        state = p
        for i in range(len(keys)):
            state = self.encrypt_one_round(state, keys[i], self.get_rc(i))
        return state

    @abstractmethod
    def decrypt_one_round(self, c, k, rc=None):
        pass

    def decrypt(self, c, keys):
        state = c
        for i in range(len(keys) - 1, -1, -1):
            state = self.decrypt_one_round(state, keys[i], self.get_rc(i))
        return state

    @abstractmethod
    def data_aug(self, c, p=None, variant=1):
        pass

    def get_rc(self, r):
        return None

    def draw_keys(self, n_samples):
        if self.use_key_schedule:
            bytes_per_word = self.bytes_per_word(self.main_key_word_size)
            main_key = np.frombuffer(
                urandom(self.n_main_key_words * bytes_per_word * n_samples), dtype=self.main_key_word_dtype
            ).reshape(self.n_main_key_words, n_samples)
            if self.main_key_word_size < 8:
                # Note: If the word size is greater than 8, it will always fit the dtype for the ciphers we use
                main_key = np.right_shift(main_key, 8 - self.main_key_word_size)
            return self.key_schedule(main_key)
        else:
            bytes_per_word = self.bytes_per_word(self.round_key_word_size)
            round_keys = np.frombuffer(
                urandom(self.n_rounds * self.n_round_key_words * bytes_per_word * n_samples),
                dtype=self.round_key_word_dtype
            ).reshape(self.n_rounds, self.n_round_key_words, n_samples)
            if self.round_key_word_size < 8:
                # Note: If the word size is greater than 8, it will always fit the dtype for the ciphers we use
                round_keys = np.right_shift(round_keys, 8 - self.round_key_word_size)
            return round_keys

    def draw_plaintexts(self, n_samples):
        return self.draw_ciphertexts(n_samples)

    def draw_ciphertexts(self, n_samples):
        bytes_per_word = self.bytes_per_word(self.word_size)
        ct = np.reshape(
            np.frombuffer(urandom(bytes_per_word * self.n_words * n_samples), dtype=self.word_dtype),
            (self.n_words, n_samples)
        )
        if self.word_size < 8:
            ct = np.right_shift(ct, 8 - self.word_size)
        return ct

    @abstractmethod
    def key_schedule(self, key):
        pass

    def rol(self, x, k):
        return ((x << k) & self.mask_val) | (x >> (self.word_size - k))

    def ror(self, x, k):
        return (x >> k) | ((x << (self.word_size - k)) & self.mask_val)