import numpy as np


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


def create_tiles(bitmap : np.ndarray, bitDim : int, tileset : np.ndarray, tileDim : int, tset : int = 0) -> list[Tile]:
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