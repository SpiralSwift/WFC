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


def extract_tile(x : int, y : int, source : np.ndarray, dim : int) -> np.ndarray:
    '''
    slices out tile at coordinates from tileset
    '''
    x0 = x * dim
    x1 = x0 + dim
    y0 = y * dim
    y1 = y0 + dim
    return np.asarray(source[x0:x1,y0:y1,:])