import numpy as np
from GAIN_DETAIL import AbstractCipher

const0 = [0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,1]
const1 = [0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0]
const2 = [1,0,1,0,0,1,0,0,0,0,1,1,0,1,0,1]
const3 = [0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,1]
const4 = [0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1]
const5 = [1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0]
const6 = [0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0]
const7 = [0,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0]
const8 = [1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1]
const9 = [0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0]
const10 = [0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,1]
const11 = [0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0]
const12 = [0,1,0,1,0,0,0,1,0,0,1,1,0,0,0,0]
const13 = [1,1,1,1,1,0,0,0,1,1,0,0,1,0,1,0]
const14 = [1,1,0,1,1,1,1,1,1,0,0,1,0,0,0,0]
const15 = [0,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1]
const16 = [0,0,0,1,1,1,0,0,0,0,1,0,0,1,0,0]
const17 = [0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0]
const18 = [0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,0]
const = np.vstack((const0,const1,const2,const3,const4,const5,const6,const7,const8,const9,const10,const11,const12,const13,const14,const15,const16,const17,const18))

p0 = [4,1,6,3,0,5,2,7]
p1 = [1,6,7,0,5,2,3,4]
p2 = [2,3,4,1,6,7,0,5]
p3 = [7,4,1,2,3,0,5,6]
P = np.vstack((p0,p1,p2,p3))

p0_inv = [4,1,6,3,0,5,2,7]
p1_inv = [3,0,5,6,7,4,1,2]
p2_inv = [6,3,0,1,2,7,4,5]
p3_inv = [5,2,3,4,1,6,7,0]
P_inv = np.vstack((p0_inv,p1_inv,p2_inv,p3_inv))

class Midori128128(AbstractCipher):

    def __init__(self, n_rounds=32, use_key_schedule=True):
        super(Midori128128, self).__init__(
            n_rounds, word_size=8, n_words=16, n_main_key_words=16, n_round_key_words=16,
            use_key_schedule=use_key_schedule,main_key_word_size = 8
        )

    # Index:             0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f
    SBOX    = np.array([0x1, 0x0, 0x5, 0x3, 0xe, 0x2, 0xf, 0x7, 0xd, 0xa, 0x9, 0xb, 0xc, 0x8, 0x4, 0x6])
    SBOXINV = np.array([0x3, 0x4, 0x6, 0x8, 0xC, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf])

    @staticmethod
    def convert_to_binary2(arr, n_words, word_size):
        sample_len = n_words * word_size
        n_samples = len(arr[0])
        x = np.zeros((sample_len, n_samples), dtype=np.uint8)
        for i in range(sample_len):
            index = i // word_size
            offset = word_size - (i % word_size) - 1
            x[i] = (arr[index] >> offset) & 1
        return x

    @staticmethod
    def convert_to_binary3(arr, word_size):
        sample_len = word_size
        n_samples = len(arr)
        x = np.zeros((sample_len, n_samples), dtype=np.uint8)
        for i in range(sample_len):
            offset = word_size - (i % word_size) - 1
            x[i] = (arr >> offset) & 1
        return x

    def substitute(self,state, n_words, word_size):
        result = []
        for k in range(n_words):
            x = self.convert_to_binary3(state[k], word_size)
            result0 = []
            result1 = []
            result2 = []
            result3 = []
            result4 = []
            for s in range(word_size):
                result0.append(x[P[k % 4][s]])
            result1.append(self.SBOX[(result0[0] << 3) + (result0[1] << 2) + (result0[2] << 1) + (result0[3])])
            result2.append(self.SBOX[(result0[4] << 3) + (result0[5] << 2) + (result0[6] << 1) + (result0[7])])
            result1 = np.array(result1[0])
            result2 = np.array(result2[0])
            x1 = self.convert_to_binary3(result1, int(word_size / 2))
            x2 = self.convert_to_binary3(result2, int(word_size / 2))
            result3 = (np.vstack((np.array(x1), np.array(x2))))
            for s in range(word_size):
                result4.append(result3[P_inv[k % 4][s]])
            result.append(
                (result4[0] << 7) + (result4[1] << 6) + (result4[2] << 5) + (result4[3] << 4) + (result4[4] << 3) + (
                result4[5] << 2) + (result4[6] << 1) + (result4[7]))
        return result

    def substitution_layer(self, state):
        return self.substitute(state,self.n_words,self.word_size)

    def inv_substitution_layer(self, state):
        return self.substitute(state, self.SBOXINV)

    @staticmethod
    def shuffle_cell(state):
        state[0] = state[0]
        temp_s1 = np.copy(state[1])
        temp_s2 = np.copy(state[2])
        temp_s3 = np.copy(state[3])
        temp_s6 = np.copy(state[6])
        state[1] = state[10]
        state[10] = state[12]
        state[12] = state[7]
        state[7] = temp_s1
        state[2] = state[5]
        state[5] = state[4]
        state[4] = state[14]
        state[14] = temp_s2
        state[3] = state[15]
        state[15] = state[8]
        state[8] = state[9]
        state[9] = temp_s3
        state[6] = state[11]
        state[11] = temp_s6
        return state

    @staticmethod
    def mix_columns(state):
        for i in [0,4,8,12]:
            state[i] = state[i] ^ state[2 + i] ^ state[3 + i]
        for i in [1,5,9,13]:
            state[i] = state[i - 1] ^ state[i] ^ state[i + 1]
        for i in [2,6,10,14]:
            state[i] = state[i - 1] ^ state[i] ^ state[i + 1]
        for i in [3,7,11,15]:
            state[i] = state[i - 3] ^ state[i - 2] ^ state[i]
        return [state[12], state[13], state[14], state[15],
                state[0], state[1], state[2], state[3],
                state[4], state[5], state[6], state[7],
                state[8], state[9], state[10], state[11]]

    @staticmethod
    def add_tweak_key(state, tweakKey):
        for i in range(16):
            state[i] = state[i] ^ tweakKey[i]
        return state



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
        s = self.shuffle_cell(s)
        s = self.mix_columns(s)
        s = self.add_tweak_key(s, k)
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
        ks = [[key[i] ^ const[0][i] for i in range(16)]]
        for i in range(self.n_rounds - 1):
            ks.append([key[j] ^ const[i + 1][j] for j in range(16)])
        return ks