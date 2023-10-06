import numpy as np
from GAIN_DETAIL import AbstractCipher


class Skinny(AbstractCipher):

    def __init__(self, n_rounds=32, use_key_schedule=True):
        super(Skinny, self).__init__(
            n_rounds, word_size=4, n_words=16, n_main_key_words=16, n_round_key_words=16,
            use_key_schedule=use_key_schedule
        )

    # Index:             0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f
    SBOX    = np.array([0xC, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf])
    SBOXINV = np.array([0x3, 0x4, 0x6, 0x8, 0xC, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf])

    @staticmethod
    def substitute(state, sb):
        result = []
        for s in state:
            result.append(sb[s])
        return result

    def substitution_layer(self, state):
        return self.substitute(state, self.SBOX)

    def inv_substitution_layer(self, state):
        return self.substitute(state, self.SBOXINV)

    @staticmethod
    def add_constants(state, constants):
        state[0] = state[0] ^ (constants & 0xf)
        state[4] = state[4] ^ (constants >> 4)
        state[8] = state[8] ^ 0x2
        return state

    @staticmethod
    def add_tweak_key(state, tweakKey):
        for i in range(8):
            state[i] = state[i] ^ tweakKey[i]
        return state

    @staticmethod
    def shift_rows(state):
        return [state[0], state[1], state[2], state[3],
                state[7], state[4], state[5], state[6],
                state[10], state[11], state[8], state[9],
                state[13], state[14], state[15], state[12]]

    @staticmethod
    def inv_shift_rows(state):
        return [state[0], state[1], state[2], state[3],
                state[5], state[6], state[7], state[4],
                state[10], state[11], state[8], state[9],
                state[15], state[12], state[13], state[14]]

    @staticmethod
    def mix_columns(state):
        for i in range(4):
            state[4 + i] = state[4 + i] ^ state[8 + i]
        for i in range(4):
            state[8 + i] = state[8 + i] ^ state[i]
        for i in range(4):
            state[12 + i] = state[12 + i] ^ state[8 + i]
        return [state[12], state[13], state[14], state[15],
                state[0], state[1], state[2], state[3],
                state[4], state[5], state[6], state[7],
                state[8], state[9], state[10], state[11]]

    @staticmethod
    def inv_mix_columns(state):
        state = [state[4], state[5], state[6], state[7],
                 state[8], state[9], state[10], state[11],
                 state[12], state[13], state[14], state[15],
                 state[0], state[1], state[2], state[3]]
        for i in range(4):
            state[12 + i] = state[12 + i] ^ state[8 + i]
        for i in range(4):
            state[8 + i] = state[8 + i] ^ state[i]
        for i in range(4):
            state[4 + i] = state[4 + i] ^ state[8 + i]
        return state

    def encrypt_one_round(self, p, k, rc=None):
        if rc is None:
            raise Exception("ERROR: Round constant has to be set for Skinny encryption")
        s = self.substitution_layer(p)
        s = self.add_constants(s, rc)
        s = self.add_tweak_key(s, k)
        s = self.shift_rows(s)
        s = self.mix_columns(s)
        return s

    def decrypt_one_round(self, c, k, rc=None):
        if rc is None:
            raise Exception("ERROR: Round constant has to be set for Skinny decryption")
        s = self.inv_mix_columns(c)
        s = self.inv_shift_rows(s)
        s = self.add_tweak_key(s, k)
        s = self.add_constants(s, rc)
        s = self.inv_substitution_layer(s)
        return s

    def data_aug(self, c, p=None, variant=1):
        if variant == 3:
            return c
        s = self.inv_mix_columns(c)
        s = self.inv_shift_rows(s)
        if variant == 2:
            return s
        if variant == 1:
            s = self.add_constants(s, self.get_rc(self.n_rounds - 1))
            for i in range(8, 16):
                s[i] = self.SBOXINV[s[i]]
            return s
        raise Exception(f'ERROR: Variant {variant} of calculating back is not implemented')

    def get_rc(self, r):
        constant = 0x1
        for key in range(r):
            constant = ((constant << 1) & 0x3f) ^ ((constant >> 5) & 1) ^ ((constant >> 4) & 1) ^ 1
        return constant

    def key_schedule(self, key):
        ks = [key[:8]]
        for i in range(self.n_rounds - 1):
            key = [key[i] for i in [9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]]
            # Add first two rows as round (tweak-)key
            ks.append(key[:8])
        return ks