import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation as animation
import random

def iterate(Z):
    # http://www.labri.fr/perso/nrougier/from-python-to-numpy/code/game_of_life_numpy.py
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
         Z[1:-1, 0:-2]                 + Z[1:-1, 2:] +
         Z[2:  , 0:-2] + Z[2:  , 1:-1] + Z[2:  , 2:])
    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)
    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1
    return Z

def update_frame(frameNum, img, gol):
    # http://www.labri.fr/perso/nrougier/from-python-to-numpy/code/game_of_life_numpy.py
    N = (gol[0:-2, 0:-2] + gol[0:-2, 1:-1] + gol[0:-2, 2:] +
         gol[1:-1, 0:-2]                 + gol[1:-1, 2:] +
         gol[2:  , 0:-2] + gol[2:  , 1:-1] + gol[2:  , 2:])
    birth = (N == 3) & (gol[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (gol[1:-1, 1:-1] == 1)
    gol[...] = 0
    gol[1:-1, 1:-1][birth | survive] = 1
    img.set_data(gol)
    return img

def display(grid):
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_frame, fargs=(img, grid,),
                                    frames=10,
                                    interval=50,
                                    save_count=50)

    plt.show()

