import numpy as np
from GAIN_DETAIL import AbstractCipher

const0 = [0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1]
const1 = [0,1,1,0,1,0,1,0,1,0,0,0,1,0,0,0]
const2 = [1,0,0,0,0,1,0,1,1,0,1,0,0,0,1,1]
const3 = [0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1]
const4 = [0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,1]
const5 = [1,0,0,0,1,0,1,0,0,0,1,0,1,1,1,0]
const6 = [0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0]
const7 = [0,1,1,1,0,0,1,1,0,1,0,0,0,1,0,0]
const8 = [1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1]
const9 = [0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0]
const10 = [0,0,1,0,1,0,0,1,1,0,0,1,1,1,1,1]
const11 = [0,0,1,1,0,0,0,1,1,1,0,1,0,0,0,0]
const12 = [0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0]
const13 = [1,1,1,1,1,0,1,0,1,0,0,1,1,0,0,0]
const14 = [1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0]
const15 = [0,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1]
const16 = [0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,0]
const17 = [0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0]
const18 = [0,0,1,1,1,0,0,0,1,1,0,1,0,0,0,0]
const = np.vstack((const0,const1,const2,const3,const4,const5,const6,const7,const8,const9,const10,const11,const12,const13,const14,const15,const16,const17,const18))

class Midori64128(AbstractCipher):

    def __init__(self, n_rounds=32, use_key_schedule=True):
        super(Midori64128, self).__init__(
            n_rounds, word_size=4, n_words=16, n_main_key_words=32, n_round_key_words=16,
            use_key_schedule=use_key_schedule
        )

    # Index:             0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f
    SBOX    = np.array([0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6])
    SBOXINV = np.array([0xc, 0xa, 0xd, 0xd, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6])

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
    def shuffle_cell(state):
        state[0] = state[0]
        temp_s1 = np.copy(state[1])
        temp_s2 = np.copy(state[2])
        temp_s3 = np.copy(state[3])
        temp_s9 = np.copy(state[9])
        state[1] = state[11]
        state[11] = state[8]
        state[8] = state[5]
        state[5] = temp_s1
        state[2] = state[6]
        state[6] = state[12]
        state[12] = state[15]
        state[15] = temp_s2
        state[3] = state[13]
        state[13] = state[4]
        state[4] = state[10]
        state[10] = temp_s3
        state[7] = state[7]
        state[9] = state[14]
        state[14] = temp_s9
        return state

    @staticmethod
    def mix_columns(state):
        for i in range(4):
            state[i] = state[i] ^ state[8 + i] ^ state[12 + i]
        for i in range(4):
            state[4 + i] = state[i] ^ state[4 + i] ^ state[8 + i]
        for i in range(4):
            state[8 + i] = state[4 + i] ^ state[8 + i] ^ state[12 + i]
        for i in range(4):
            state[12 + i] = state[i] ^ state[4 + i] ^ state[12 + i]
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
    def inv_shift_rows(state):
        return [state[0], state[1], state[2], state[3],
                state[5], state[6], state[7], state[4],
                state[10], state[11], state[8], state[9],
                state[15], state[12], state[13], state[14]]



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
            if (i + 1) % 2 == 1:
                ks.append([key[j] ^ const[i + 1][j - 16] for j in range(16, 32)])
            else:
                ks.append([key[j] ^ const[i + 1][j] for j in range(16)])
        return ks
