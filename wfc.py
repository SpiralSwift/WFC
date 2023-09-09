import io
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib import animation


def extract_tile(x : int, y : int, source : np.ndarray, dim : int) -> np.ndarray:
    '''
    slices out tile at coordinates from tileset
    '''
    x0 = x * dim
    x1 = x0 + dim
    y0 = y * dim
    y1 = y0 + dim
    
    return np.asarray(source[x0:x1,y0:y1,:])


def create_bitmask(tile : np.ndarray, dim : int) -> np.ndarray:
    '''
    Interpret bitmap data to create bitmask
    uses RGB channels separately
    '''
    mask = np.zeros((dim,dim),dtype=int)
    # check each pixel in tile
    for x in range(dim):
        for y in range(dim):
            # check if any color channel nonzero &assign using first one; ignore alpha channel
            for i in range(3):
                if tile[x,y,i] > 0:
                    mask[x,y] = i+1
                    break
    mask = mask.reshape((mask.size,))

    #print(np.sum(tile,2))
    #print(mask)
    return mask


def get_neighbor_coords(x : int, y : int, dim : list[int]) -> np.ndarray:
    '''
    get coordinates of valid tiles neighboring tile at given position
    first two values: coordinates
    second two values: delta from reference position
    '''
    coords = []
    if x > 0 : coords.append([x-1,y,-1,0])
    if x < dim[0]-1 : coords.append([x+1,y,1,0])
    if y > 0 : coords.append([x,y-1,0,-1])
    if y < dim[1]-1 : coords.append([x,y+1,0,1])
    return np.asarray(coords)
    return np.asarray(coords)


class Tile:
    def __init__(self, img : np.ndarray, bitTile : np.ndarray, name : str) -> None:
        self.img = img
        self.bitTile = bitTile
        self.name = name

        self.bitmask = create_bitmask(bitTile,len(bitTile))
        self.mask = self.bitmask[4] # "native" mask layer


def create_tiles(bitmap : np.ndarray, bitDim : int, tileset : np.ndarray, tileDim : int) -> list[Tile]:
        # get dimensions of tileset / bitmap in tiles
        dimx = int(bitmap.shape[0] / bitDim)
        dimy = int(bitmap.shape[1] / bitDim)

        # create tiles
        tiles = []
        for x in range(dimx):
            for y in range(dimy):
                bitTile = extract_tile(x,y,bitmap,bitDim)[:,:,:3] # slice out alpha values
                if np.sum(bitTile) > 0: # check if blank space
                    imgTile = extract_tile(x,y,tileset,tileDim)
                    name = str(x)+'|'+str(y)
                    tile = Tile(imgTile, bitTile, name)
                    tiles.append(tile)
        
        return tiles


class Board:
    def __init__(self, dim : list[int], tiles : list[Tile], nullidx : int = 0, noneidx = -1) -> None:
        self.dim = dim
        self.tiles = tiles

        self.nullidx = nullidx
        self.noneidx = noneidx

        self.tileDim = tiles[0].img.shape[0]
        self.bitDim = tiles[0].bitTile.shape[0]
        self.stateDim = len(tiles)

        # TODO: generate these
        self.cornerbits = np.asarray([1,0,1,0,0,0,1,0,1])
        self.edgebits = np.asarray([0,1,0,1,0,1,0,1,0])
        self.centrebit = 4

        self.board = np.ones(dim,dtype=int) * noneidx # array of indices of tiles in list
        self.mask = np.ones(dim,dtype=int) * nullidx # array of masks for tiles
        self.img = np.zeros((dim[0]*tileDim,dim[1]*tileDim,4))
        self.states = np.ones((dim[0],dim[1],self.stateDim),dtype=int) # boolean matrix of valid tiles ("states") at each position

    def collapse_states(self, states : np.ndarray, dx : int, dy : int, reftileidx : int = -1) -> np.ndarray:
        '''
        reduce state space of given tile based on state of neighbor
        tiles: list of tiles
        states: list of possible states for current tile
        dx, dt: delta from reference tile to undetermined tile
        refidx: index of reference tile in tile list (tile being compared against)
        '''

        dim = 3
        testidx = (dx+1)*dim + (dy+1) # bitmask index for undetermined tile
        refidx = 8 - testidx # bitmask index for ref tile
        step = 1 if dy == 0 else dim # accomodates directionality of edge

        # process null case
        if reftileidx == -1:
            return np.copy(states) # empty tiles can bind to anything

        # collapse state space of tile
        newstates = np.zeros(states.shape,dtype=int)
        for i in np.nonzero(states)[0]:
            # compare tile edges by looking at corresponding mask values and values on either side along edge
            query = True
            for j in range(dim):
                tidx = testidx + step * (j-1)
                ridx = refidx + step * (j-1)
                if self.tiles[i].bitmask[tidx] != self.tiles[reftileidx].bitmask[ridx]:
                    query = False
                    break
            if query : newstates[i] = 1
        return newstates

    def _set_tile(self, idx : int, x : int, y : int) -> None:
        '''
        update data for tile location
        (assumes lavid coordinates!)
        '''
        tile = self.tiles[idx]

        # flag tile on board
        self.board[x,y] = idx
        self.mask[x,y] = tile.mask # mask of tile is value of central bit
        
        # update image
        x0 = x * self.tileDim
        x1 = x0 + self.tileDim
        y0 = y * self.tileDim
        y1 = y0 + self.tileDim
        self.img[x0:x1,y0:y1,:] = tile.img

        # update state space of tile
        newstates = np.zeros(len(self.tiles))
        newstates[idx] = 1
        self.states[x,y,:] = newstates

    def assign_tile(self, idx : int, x : int, y : int) -> None:
        '''
        place a specific tile
        '''
        # check location in bounds
        if not self.check_in_bounds(x,y) : return

        # update tile data
        self._set_tile(idx,x,y)

        # update state space of neighbors
        self.update_neighbors(idx,x,y)

    def update_tile(self, x : int, y : int) -> None:
        '''
        automatically assign a variant to a tile
        '''
        # check location in bounds
        if not self.check_in_bounds(x,y) : return
        
        # update state space based on states of neighbors
        newstates = self.states[x,y,:]
        coords = get_neighbor_coords(x,y,self.dim)
        for pos in coords:
            xx = pos[0] # position of neighbor
            yy = pos[1]
            dx = pos[2] # delta to neighbor from current
            dy = pos[3]

            newstates = self.collapse_states(newstates,dx,dy,self.board[xx,yy])
        self.states[x,y,:] = newstates


        # randomly assign state
        #print(np.sum(newstates))
        states = np.nonzero(newstates)[0]
        if states.size == 0: # catch if no states available
            states = [self.nullidx]
            print('Out of states! Assigning null state')
        state = np.random.choice(states)
        self._set_tile(state,x,y)

        # update state space of neighbors
        self.update_neighbors(state,x,y)

    def update_neighbors(self, idx : int, x : int, y : int) -> None:
        coords = get_neighbor_coords(x,y,self.dim)
        for pos in coords:
            xx = pos[0] # position of neighbor
            yy = pos[1]
            dx = -pos[2] # delta to neighbor from current
            dy = -pos[3]
            #print(str(xx)+','+str(yy))
            states = self.states[xx,yy,:]
            self.states[xx,yy,:] = self.collapse_states(states,dx,dy,idx)
    
    def check_in_bounds(self, x : int, y : int) -> bool:
        if x < 0 or x >= self.dim[0] or y < 0 or y >= self.dim[1]:
            print('Index ('+str(x)+','+str(y)+') out of bounds!')
            return False
        return True

    def print_states(self) -> None:
         print(np.sum(self.states,2))



# --- PARAMETERS ---
tilesetPath = 'cliffs.png'
bitmapPath = 'bitmask_all.png'
bitDim = 3 # dimension of "tiles" in bitmap
tileDim = 16 # dimension of tiles in tileset
boardDim = [25,25]
seed = 2023 # seed for RNG


# --- SETUP ---
# seed RNG
#np.random.seed(seed) # not recommended, but whatever

# load images
bitmap = np.asarray(iio.imread(bitmapPath),dtype=int)
tileset = np.asarray(iio.imread(tilesetPath),dtype=int)

# initialize tiles & board
tiles = create_tiles(bitmap, bitDim, tileset, tileDim)
board = Board(boardDim, tiles)


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
    board.update_tile(x,y)
    #board.print_states()
    im.set_data((board.img).astype(np.uint8))

fig = plt.figure()
im = plt.imshow((board.img).astype(np.uint8))

nframes = (boardDim[0]-buffer*2) * (boardDim[1]-buffer*2) -offset
anim = animation.FuncAnimation(fig, animate_wfc, frames=nframes, interval=10, fargs=(board,im,buffer))
plt.show()