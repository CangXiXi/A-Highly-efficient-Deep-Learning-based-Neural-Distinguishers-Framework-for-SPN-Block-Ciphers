from train_nets import train_distinguisher
from Midori64128 import Midori64128

midori64128 = Midori64128(n_rounds=3)
train_distinguisher(
    midori64128, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], case=3, lr_high=0.0011, lr_low=0.000045,
    kernel_size=3, reg_param=0.000000043
)






















