from train_nets import train_distinguisher
from Midori128128 import Midori128128

# Script for training a Skinny distinguisher using the same hyper-parameter as in the paper

midori128128 = Midori128128(n_rounds=5)
train_distinguisher(
    midori128128, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], case=3, lr_high=0.0011, lr_low=0.000045,
    kernel_size=3, reg_param=0.000000043
)