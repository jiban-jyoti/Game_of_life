import numpy as np
from jv import fft_convolve2d
from PIL import Image
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import json

with open('conway.json') as json_file:
    data = json.load(json_file)
    #print(data['length'])
    m = (data['length'])
    n = (data['breadth'])
    seed = (data['random_seed'])
np.random.seed(seed)

def conway(state, k=None):
    """
    Conway's game of life state transition
    """

    # set up kernel if not given
    if k == None:
        m, n = state.shape
        k_arr = np.zeros((m, n))
        k_arr[m//2-1 : m//2+2, n//2-1 : n//2+2] = np.array([[1,1,1],[1,0,1],[1,1,1]])

    # computes sums around each pixel
    b = fft_convolve2d(state,k_arr).round()
    c = np.zeros(b.shape)

    c[np.where((b == 2) & (state == 1))] = 1
    c[np.where((b == 3) & (state == 1))] = 1

    c[np.where((b == 3) & (state == 0))] = 1

    # return new state
    return c




#m,n = 100,100
A = np.random.random(m*n).reshape(m, n).round()

ims = []
fig = plt.figure()

for i in range(0,400):
    img_plot = plt.imshow(A, interpolation="nearest", cmap = plt.cm.gray, animated=True)
    A = conway(A)
    im = plt.imshow(A)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
plt.show()
