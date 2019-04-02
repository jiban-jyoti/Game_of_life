import numpy as np
from jv import fft_convolve2d
from PIL import Image
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import json

with open('conway.json') as json_file:
    data = json.load(json_file)
    #print(data['length'])
    p = (data['len_grid'])
    q = (data['br_grid'])
    seed = (data['rnd_seed'])
np.random.seed(seed)

def conway(state, k=None):
    """
    Conway's game of life state transition
    """

    # setting kernel if not given
    if k == None:
        p, q = state.shape
        k_arr = np.zeros((p, q))
        k_arr[p//2-1 : p//2+2, q//2-1 : q//2+2] = np.array([[1,1,1],[1,0,1],[1,1,1]])

    # computes sum around  pixel using fft algo
    p = fft_convolve2d(state,k_arr).round()
    s = np.zeros(p.shape)

    s[np.where((p == 2) & (state == 1))] = 1
    s[np.where((p == 3) & (state == 1))] = 1

    s[np.where((p == 3) & (state == 0))] = 1

    # return new state
    return s




#m,n = 100,100
A = np.random.random(p*q).reshape(p, q).round()

ims = []
fig = plt.figure()

for i in range(0,400):
    image_plot = plt.imshow(A, interpolation="nearest", cmap = plt.cm.gray, animated=True)
    A = conway(A)
    im = plt.imshow(A)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
plt.show()
