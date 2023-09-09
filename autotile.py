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


def get_autotile_neighbors(x : int, y : int, dim : list[int]) -> np.ndarray:
    '''
    get coordinates of valid tiles neighboring tile at given position
    first two values: coordinates
    second two values: delta from reference position
    '''
    x0 = x-1 if x > 0 else x
    x1 = x+2 if x < dim[0] -1 else x+1
    y0 = y-1 if y > 0 else y
    y1 = y+2 if y < dim[1] -1 else y+1
    coords = [[xx,yy,xx-x,yy-y] for xx in range(x0,x1) for yy in range(y0,y1) if (xx != x or yy != y)]
    return np.asarray(coords)


class Tile:
    def __init__(self, img : np.ndarray, bitTile : np.ndarray, name : str, tset : int) -> None:
        self.img = img
        self.bitTile = bitTile
        self.name = name
        self.tileset = tset

        self.bitmask = create_bitmask(bitTile,len(bitTile))
        self.mask = self.bitmask[4] # "native" mask layer


def create_tiles(bitmap : np.ndarray, bitDim : int, tileset : np.ndarray, tileDim : int, tset : int) -> list[Tile]:
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
                    tile = Tile(imgTile, bitTile, name, tset)
                    tiles.append(tile)
        
        return tiles


class Board:
    def __init__(self, dim : list[int], tiles : list[list[Tile]], nullidx : int = 0, noneidx = -1) -> None:
        self.dim = dim
        self.tiles = tiles

        self.nullidx = nullidx
        self.noneidx = noneidx

        self.tileDim = tiles[0][0].img.shape[0]
        self.bitDim = tiles[0][0].bitTile.shape[0]

        # TODO: generate these
        self.cornerbits = np.asarray([1,0,1,0,0,0,1,0,1])
        self.edgebits = np.asarray([0,1,0,1,0,1,0,1,0])
        self.centrebit = 4

        self.board = np.ones(dim,dtype=int) * noneidx # array of indices of tiles in list
        self.mask = np.ones(dim,dtype=int) * nullidx # array of masks for tiles
        self.tsets = np.ones(dim,dtype=int) * noneidx # array of indices of tiles in list
        self.img = np.zeros((dim[0]*tileDim,dim[1]*tileDim,4))

    def collapse_autotile_states(self, states : np.ndarray, dx : int, dy : int, refmask = 0) -> np.ndarray:
        '''
        reduce state space of given tile based on state of neighbor
        tiles: list of tiles
        states: list of possible states for current tile
        dx, dt: delta from reference tile to undetermined tile
        refidx: index of reference tile in tile list (tile being compared against)
        '''

        # collapse state space of tile
        newstates = np.zeros(states.shape,dtype=int)
        for i in np.nonzero(states)[0]:
            testidx = (dx+1)*3 + (dy+1)
            if self.tiles[i].bitmask[testidx] == refmask : newstates[i] = 1
        return np.asarray(newstates)

    def _set_tile(self, tset : int, idx : int, x : int, y : int) -> None:
        '''
        update data for tile location
        (assumes lavid coordinates!)
        '''
        tileset = self.tiles[tset]
        tile = tileset[idx]

        # flag tile on board
        self.board[x,y] = idx
        self.mask[x,y] = tile.mask # mask of tile is value of central bit
        self.tsets[x,y] = tile.tileset
        
        # update image
        x0 = x * self.tileDim
        x1 = x0 + self.tileDim
        y0 = y * self.tileDim
        y1 = y0 + self.tileDim
        self.img[x0:x1,y0:y1,:] = tile.img

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

    def autotile(self, tset : int, x : int, y : int) -> None:
        '''
        automatically assign a variant to a tile and its neighbors
        '''
        # check location in bounds
        if not self.check_in_bounds(x,y) : return

        # set current tile
        # TODO: make this a little more robust
        self._apply_autotile(tset,x,y,self.tiles[tset][0].mask)

        #print(self.mask)
        #print()
        #return

        # update neighbors
        coords = get_autotile_neighbors(x,y,self.dim)
        for pos in coords:
            xx = pos[0] # position of neighbor
            yy = pos[1]
            if self.mask[xx,yy] != self.nullidx : self._apply_autotile(self.tsets[xx,yy],xx,yy,self.mask[xx,yy])

    def _apply_autotile(self, tset : int, x : int, y : int, mask : int = 0) -> None:
        '''
        determine set tile using autotiling
        (assumes valid grid position!)
        x,y: grid position of tile
        mask: mask of tile being placed
        '''

        # flag tile present
        if mask > 0 : self.mask[x,y] = mask

        # get local masks
        bitmask = np.ones((self.bitDim,self.bitDim),dtype=int) * self.nullidx
        for dx in range(self.bitDim):
            for dy in range(self.bitDim):
                xx = x + dx-1
                yy = y + dy-1
                if self.check_in_bounds(xx,yy):
                    bitmask[dx,dy] = self.mask[xx,yy]
        bitmask = bitmask.reshape((bitmask.size,))
        bitmask[bitmask != mask] = 0 # treat tiles with different mask from native as empty

        # compare to tile bitmasks
        diffcts = np.zeros(len(self.tiles[tset]),dtype=int)
        for i, tile in enumerate(self.tiles[tset]):
            # treat bits with different mask from native as empty
            testmask = np.copy(tile.bitmask)
            testmask[testmask != mask] = 0

            delta = bitmask - testmask
            #delta[delta > 0] = 1 # weight of missing bit
            #delta[delta < 0] = 2 # weight of extra bit
            delta[delta != 0] = 1 # weight missing & extra bits equally
            delta[self.edgebits == 1] *= 3 # weight of edge bits (higest ratio or corner differences to edge differences is 2)
            delta[self.cornerbits == 1] *= 1 # weight of edge bits
            delta[self.centrebit] *= 0 # weight of central bit
            diffcts[i] = np.sum(delta) # count total differences

        # randomly assign by selecting among closest-matching tiles
        minidx = np.where(diffcts == diffcts.min())[0] # get indices of minimum difference counts
        if len(minidx) > 1: # report imperfect matches
            print('Imperfect match!')
            print(bitmask)
            print(minidx)
            for i in minidx:
                print(self.tiles[tset][i].bitmask)
        state = np.random.choice(minidx)
        self._set_tile(tset,state,x,y)

    def _apply_autotile_legacy(self, x : int, y : int) -> None: # broken
        '''
        determine set tile using autotiling
        (assumes valid grid position!)
        '''
        # TODO: create weighted version where start from zero and coutn how many times tile valid
        # update state space based on states of neighbors
        newstates = np.ones(len(self.states[x,y]),dtype=int)
        coords = get_autotile_neighbors(x,y,self.dim)
        for pos in coords:
            xx = pos[0] # position of neighbor
            yy = pos[1]
            dx = pos[2] # delta to neighbor from current
            dy = pos[3]

            newstates = self.collapse_autotile_states(newstates,dx,dy,self.mask[xx,yy])
            print(np.sum(newstates))
        print()
        self.states[x,y,:] = newstates

        # randomly assign state
        states = np.nonzero(newstates)[0]
        if states.size == 0: # catch if no states available
            states = [self.nullidx]
            print('Out of states! Assigning null state')
        state = np.random.choice(states)
        self._set_tile(state,x,y)

    def check_in_bounds(self, x : int, y : int) -> bool:
        if x < 0 or x >= self.dim[0] or y < 0 or y >= self.dim[1]:
            print('Index ('+str(x)+','+str(y)+') out of bounds!')
            return False
        return True
    
    def print_states(self) -> None:
         print(np.sum(self.states,2))



# --- PARAMETERS ---
tilesetPaths = ['cliffs.png','snow_cliffs.png','fences.png']
bitmapPaths = ['bitmask_all.png','bitmask_all.png','bitmask_sides.png']
channels = [0,0,1]
bitDim = 3 # dimension of "tiles" in bitmap
tileDim = 16 # dimension of tiles in tileset
boardDim = [25,25]
seed = 2023 # seed for RNG


# --- SETUP ---
# seed RNG
#np.random.seed(seed) # not recommended, but whatever

# load images
ntilesets = min([len(tilesetPaths),len(bitmapPaths),len(channels)])
bitmaps = [np.asarray(iio.imread(path),dtype=int) for path in bitmapPaths]
tilesets = [np.asarray(iio.imread(path),dtype=int) for path in tilesetPaths]

# modify convert white bits to r/g/b
for i in range(ntilesets):
    idx = channels[i]
    channel = np.zeros(bitmaps[i].shape,dtype=int)
    channel[:,:,idx] = 1
    bitmaps[i] *= channel

# initialize tiles & board
tiles = [create_tiles(bitmaps[i], bitDim, tilesets[i], tileDim, i) for i in range(ntilesets)]
board = Board(boardDim, tiles)


# --- TEST ---
if False:
    id = 0#34
    board._set_tile(id,1,2)
    board._set_tile(id,2,1)
    board.autotile(2,2)

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
    board.autotile(tset[frame],x,y)
    #board.print_states()
    im.set_data((board.img).astype(np.uint8))

fig = plt.figure()
im = plt.imshow((board.img).astype(np.uint8))

nframes = (boardDim[0]-buffer*2) * (boardDim[1]-buffer*2) -offset
anim = animation.FuncAnimation(fig, animate_autotile, frames=nframes, interval=10, fargs=(board,im,buffer,xy))
plt.show()