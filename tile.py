import numpy as np


class Tileset:
    '''
    Collection of tiling-based sprite variants for a tile of a given type

    tiledim: dimensions of tile sprites
    nbits: number of bits in tile bitmask (number of neighbors used in tiling rules)
    mask: indicates which "class" of tiles this tile belongs to; indicates which other tilesets to tile with
    bitmasks: 2D array of bitmasks for each tile  (ntiles x nbits)
    sprites: array of sprites for each tile (ntiles x ...)
    '''
    def __init__(self, image : np.ndarray, tiledim : int, bitmap : np.ndarray, bitdim : int, mask : int = 1) -> None:
        '''
        image: spritesheet with all tiles
        tiledim: dimensions of tile in spritesheet
        bitmap: image containing tiling information
        bitdim: dimensions of tile in bitmap
        mask: indicates which other tilesets this set can tile with
        TODO: allow multiple masks & asymmetrical tiling rules ?
        '''
        self.mask = mask
        self.tiledim = tiledim # assumes square tiles
        self.bitdim = bitdim
        self.nbits = bitdim ** 2 # assumes square bitmaps

        # extract tile sprites and masks
        dim = tuple([int(bitmap.shape[i] / bitdim) for i in range(2)])  # get dimensions of tileset & bitmap in tiles
        self.bitmasks = [] # bitmaps for each tile
        self.sprites = [] # sprites for each tile
        for x, y in np.ndindex(dim):
            bitTile = extract_tile(x,y,bitmap,bitdim)[:,:,:3] # get rgb values
            if np.sum(bitTile) > 0: # check if blank space
                # create tile bitmask
                # TODO: assign meanings to different colors (non-binary bitmask)
                bitmask = np.sum(bitTile,axis=2) # check for any nonzero pixel
                bitmask[bitmask > 0] = 1
                bitmask = bitmask.reshape((self.nbits,)) # convert to 1D array

                # extract sprite and store in array
                sprite = extract_tile(x,y,image,tiledim)
                self.sprites.append(sprite)
                self.bitmasks.append(bitmask)
        self.bitmasks = np.array(self.bitmasks)
        self.sprites = np.array(self.sprites)


class Tile:
    def __init__(self, img : np.ndarray, bitTile : np.ndarray, name : str, tset : int) -> None:
        self.img = img
        self.bitTile = bitTile
        self.name = name
        self.tileset = tset

        self.bitmask = create_bitmask(bitTile,len(bitTile))
        self.mask = self.bitmask[4] # "native" mask layer


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


def get_neighbor_coords(x : int, y : int, dim : list[int]) -> np.ndarray:
    '''
    legacy version of get_autotile_neighbors()
    '''
    coords = []
    if x > 0 : coords.append([x-1,y,-1,0])
    if x < dim[0]-1 : coords.append([x+1,y,1,0])
    if y > 0 : coords.append([x,y-1,0,-1])
    if y < dim[1]-1 : coords.append([x,y+1,0,1])
    return np.asarray(coords)


def create_tiles(bitmap : np.ndarray, bitdim : int, tileset : np.ndarray, tiledim : int, tset : int = 0) -> list[Tile]:
        # get dimensions of tileset / bitmap in tiles
        dimx = int(bitmap.shape[0] / bitdim)
        dimy = int(bitmap.shape[1] / bitdim)

        # create tiles
        tiles = []
        for x in range(dimx):
            for y in range(dimy):
                bitTile = extract_tile(x,y,bitmap,bitdim)[:,:,:3] # slice out alpha values
                if np.sum(bitTile) > 0: # check if blank space
                    imgTile = extract_tile(x,y,tileset,tiledim)
                    name = str(x)+'|'+str(y)
                    tile = Tile(imgTile, bitTile, name, tset)
                    tiles.append(tile)

        return tiles