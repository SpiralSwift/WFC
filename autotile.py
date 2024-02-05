import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib import animation
from tile import Tileset
from board import Board


if __name__ == "__main__":
    # --- PARAMETERS ---
    tilesetPaths = ['cliffs.png','snow_cliffs.png','fences.png']
    bitmapPaths = ['bitmask_all.png','bitmask_all.png','bitmask_sides.png']
    masks = [0,0,1] # sets which tilesets wll tile together
    bitdim = 3 # dimension of "tiles" in bitmap
    tiledim = 16 # dimension of tiles in tileset
    boardDim = [25,25] # dimensions of board (in tiles)
    seed = 2024 # seed for RNG


    # --- SETUP ---
    # seed RNG
    #np.random.seed(seed) # not recommended, but whatever

    # load images
    ntilesets = min([len(tilesetPaths),len(bitmapPaths),len(masks)])
    bitmaps = [np.asarray(iio.imread(path),dtype=int) for path in bitmapPaths]
    sprites = [np.asarray(iio.imread(path),dtype=int) for path in tilesetPaths]

    # initialize tiles & board
    tiles = [Tileset(sprites[i],tiledim,bitmaps[i],bitdim,masks[i]+1) for i in range(ntilesets)]
    board = Board(boardDim, tiles)


    # --- TEST ---
    if False:
        tset = 0
        board._set_tile(1,2,tset,1)
        board._set_tile(2,1,tset,2)
        board.autotile(2,2,tset)

        fig = plt.figure()
        im = plt.imshow((board.img).astype(np.uint8))
        plt.show()
        exit()

    # --- SIMULATION ---
    buffer = 1
    offset = 0
    nvals = (board.dim[0] - 2*buffer) * (board.dim[1] - 2*buffer)
    xy = np.arange(0,nvals)
    np.random.shuffle(xy)
    tset = np.random.randint(ntilesets,size=nvals)

    def animate_autotile(frame, board : Board, im, buffer, xy : np.ndarray):
        x = xy[frame] // (board.dim[0]-buffer*2) + buffer
        y = xy[frame] % (board.dim[1]-buffer*2) + buffer
        board.autotile(x,y,tset[frame])
        #board.print_states()
        im.set_data((board.img).astype(np.uint8))

    fig = plt.figure()
    im = plt.imshow((board.img).astype(np.uint8))

    nframes = (boardDim[0]-buffer*2) * (boardDim[1]-buffer*2) -offset
    anim = animation.FuncAnimation(fig, animate_autotile, frames=nframes, interval=20, fargs=(board,im,buffer,xy))
    plt.show()