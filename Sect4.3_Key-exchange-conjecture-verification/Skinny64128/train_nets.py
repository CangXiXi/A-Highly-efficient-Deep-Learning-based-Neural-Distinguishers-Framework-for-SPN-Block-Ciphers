import numpy as np
from pickle import dump
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from make_train_data import make_train_data
bs = 5000
wdir = './'


def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)


def make_checkpoint(file):
    return ModelCheckpoint(file, monitor='val_loss', save_best_only=True)

def make_resnet(
        num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=5, reg_param=0.0001,
        final_activation='sigmoid', cconv=False
):
    Conv = Conv2D
    inp = Input(shape=(num_blocks * word_size * 3,))
    rs = Reshape((3, word_size, num_blocks))(inp)
    perm = Permute((1, 3, 2))(rs)
    conv0 = Conv(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return model


def train_distinguisher(
        cipher, diff, n_train_samples=10**7, n_val_samples=10**6, n_epochs=30, depth=10, n_neurons=64, kernel_size=3,
        n_filters=32, reg_param=10 ** -5, lr_high=0.002, lr_low=0.0001, cconv=False, case=0
):
    n_rounds = cipher.get_n_rounds()
    cipher_name = type(cipher).__name__
    result_base_name = f'{wdir}{cipher_name}_{n_rounds}_best_'
    net = make_resnet(
        depth=depth, d1=n_neurons, d2=n_neurons, ks=kernel_size, num_filters=n_filters, reg_param=reg_param,
        cconv=cconv, word_size=cipher.get_word_size(), num_blocks=cipher.get_n_words()
    )
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    X, Y = make_train_data(n_train_samples, cipher, diff, case)
    X_eval, Y_eval = make_train_data(n_val_samples, cipher, diff, case)
    check = make_checkpoint(f'{result_base_name}.h5')
    lr = LearningRateScheduler(cyclic_lr(10, lr_high, lr_low))
    h = net.fit(X, Y, epochs=n_epochs, batch_size=bs, validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    np.save(wdir+'h'+str(n_rounds)+'r_depth'+str(depth)+'val_acc'+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(n_rounds)+'r_depth'+str(depth)+'val_loss'+'.npy', h.history['val_loss']);
    np.save(wdir+'h'+str(n_rounds)+'r_depth'+str(depth)+'loss'+'.npy', h.history['loss']);
    np.save(wdir+'h'+str(n_rounds)+'r_depth'+str(depth)+'acc'+'.npy', h.history['acc']);
    dump(h.history,open(wdir+'hist'+str(n_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return net, h


