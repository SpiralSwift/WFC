import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib import animation
from tile import Tileset
from board import Board


if __name__ == "__main__":
    # --- PARAMETERS ---
    tilesetPath = 'cliffs.png'
    bitmapPath = 'bitmask_all.png'
    bitdim = 3 # dimension of "tiles" in bitmap
    tiledim = 16 # dimension of tiles in tileset
    boarddim = [25,25]
    seed = 2024 # seed for RNG


    # --- SETUP ---
    # seed RNG
    #np.random.seed(seed) # not recommended, but whatever

    # load images
    bitmap = np.asarray(iio.imread(bitmapPath),dtype=int)
    sprites = np.asarray(iio.imread(tilesetPath),dtype=int)

    # initialize tiles & board
    tileset = Tileset(sprites,tiledim,bitmap,bitdim)
    board = Board(boarddim, [tileset])


    # --- TEST ---
    if False:
        id = 0#34

        fig = plt.figure()
        im = plt.imshow((board.img).astype(np.uint8))
        plt.show()
        exit()

    # --- SIMULATION ---
    buffer = 1
    offset = 0

    def animate_wfc(frame, board : Board, im, buffer):
        x = frame // (board.dim[0]-buffer*2) + buffer
        y = frame % (board.dim[1]-buffer*2) + buffer
        board.collapse_tile(x,y)
        #board.print_states()
        im.set_data((board.img).astype(np.uint8))

    fig = plt.figure()
    im = plt.imshow((board.img).astype(np.uint8))

    nframes = (boarddim[0]-buffer*2) * (boarddim[1]-buffer*2) -offset
    anim = animation.FuncAnimation(fig, animate_wfc, frames=nframes, interval=10, fargs=(board,im,buffer))
    plt.show()