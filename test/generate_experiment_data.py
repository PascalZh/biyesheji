#!/usr/bin/env python3
import argparse
FILENAME = "edge_detection_dataset"
NUM_NOISE = 10

parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--show",
                    action="store_true",
                    help="show " + FILENAME + " and exit")
parser.add_argument("-n", "--num_noise", default=20, help="set NUM_NOISE")
args = parser.parse_args()

import sys
import numpy as np
from numpy import array, arange, concatenate
from numpy.random import randn
import matplotlib.pyplot as plt
sys.path.append('..')
from utils import save_folders, load_folders


def poly2_0(x):
    return 2 + 0.1 * x**2


func = poly2_0

if args.show:
    for xy in load_folders(FILENAME, func.__name__):
        plt.plot(xy[:, 0], xy[:, 1], 'g.')
        plt.show()
    sys.exit()

NUM_NOISE = args.num_noise

x = arange(0, 10, 0.1)
for i in range(4):
    x = concatenate((x, x + randn(len(x)) / 10))

y = func(x)
gaussian_noise = randn(len(x)) / 2
y += gaussian_noise

plt.subplot(121)
plt.plot(x, y, 'g.')
plt.title("add gaussian noise")

thresh = .5
xy = np.array([[x_, y_] for x_, y_ in zip(x, y) if func(x_) - y_ < thresh])
x = xy[:, 0]
y = xy[:, 1]

plt.subplot(122)
plt.plot(x, y, 'g.')
plt.title(f"remove gaussian noise \nthat are {thresh} lower than true value\n"
          f"please click {NUM_NOISE} times to add noise")
# click 10 times to add noise
# plt.get_current_fig_manager().full_screen_toggle()
pos = plt.ginput(NUM_NOISE)
xy = concatenate((xy, array(pos)), axis=0)

plt.show()

print(f"xy of shape {xy.shape} saved")
save_folders(FILENAME, func.__name__, xy)
