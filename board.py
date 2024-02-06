import numpy as np
import tile


NULLIDX = -1

class Board:
    def __init__(self, dim : tuple[int,int], tilesets : list[tile.Tileset]) -> None:
        self.dim = dim
        self.tilesets = tilesets

        # assumes uniform values across tilesets
        self.tiledim = tilesets[0].tiledim
        self.nbits = tilesets[0].nbits
        self.bitdim = int(np.sqrt(self.nbits)) # assumes square neighbor rules

        # TODO: generate these ?
        cornerbits = np.asarray([1,0,1,0,0,0,1,0,1])
        edgebits = np.asarray([0,1,0,1,0,1,0,1,0])
        centrebit = 4
        self.weights = edgebits * 3 + cornerbits # highest ratio of corner differences to edge differences is 2
        self.weights[centrebit] = 0 # ignore central bit

        # initialize tile layers
        self.tile = np.ones(dim,dtype=int) * NULLIDX # indicates index of each tile in tileset
        self.tset = np.ones(dim,dtype=int) * NULLIDX # indicates which tileset each tile belongs to
        self.mask = np.zeros(dim,dtype=int) # indicates which mask each tile belongs to (0 -> empty)
        self.img = np.zeros((dim[0]*self.tiledim,dim[1]*self.tiledim,4)) # board image

        # create emoty tile image
        self.emptyTile = np.zeros((self.tiledim,self.tiledim,4))
        self.emptyTile[:,:,3] = 1 # [0,0,0,1] -> black

    def _set_tile(self, x : int, y : int, tset : int, idx : int) -> None:
        '''
        Set tile at given location
        NOTE: assumes valid coordinates
        x, y: coordinates
        tset: tileset index
        idx: tile index in tileset
        '''
        tileset = self.tilesets[tset]

        # flag tile on board
        self.tile[x,y] = idx
        self.tset[x,y] = tset
        self.mask[x,y] = tileset.mask
        
        # update image
        x0 = x * self.tiledim
        x1 = x0 + self.tiledim
        y0 = y * self.tiledim
        y1 = y0 + self.tiledim
        self.img[x0:x1,y0:y1,:] = tileset.sprites[idx]

    def _remove_tile(self, x : int, y : int) -> None:
        '''
        Remove tile at given location
        NOTE: assumes valid coordinates
        '''
        # remove tile flags
        self.tile[x,y] = NULLIDX
        self.tset[x,y] = NULLIDX
        self.mask[x,y] = 0
        
        # update image
        x0 = x * self.tiledim
        x1 = x0 + self.tiledim
        y0 = y * self.tiledim
        y1 = y0 + self.tiledim
        self.img[x0:x1,y0:y1,:] = self.emptyTile


    def collapse_tile(self, x : int, y : int, tset : int = 0) -> None:
        '''
        Use WFC to assign a tile based on its neighbors
        NOTE: designed for use with a single tileset
        '''
        tileset = self.tilesets[tset]
        
        # determine required bitmask
        bitmask = self._get_enforced_bitmask(x,y)
        delta = np.abs(tileset.bitmasks - bitmask)
        delta = np.nansum(delta,axis=1)

        # get valid tile
        print(bitmask)
        minidx = np.where(delta == delta.min())[0] # list of all minima
        if min(delta) > 0: # report imperfect matches
            print(f'Imperfect match! Candidates: {bitmask}')
            for i in minidx : print(f'   {tileset.bitmasks[i,:]} ({delta[i]})')

        # choose a random tile
        idx = np.random.choice(minidx)
        self._set_tile(x,y,tset,idx)
        


    def _get_enforced_bitmask(self, x : int, y : int) -> np.ndarray:
        '''
        Get local bitmask based corresponding bits in bitmasks of neighbors (rather than their actual masks)
        NaN -> no value enforced
        '''
        bitmask = np.full((self.bitdim,self.bitdim),np.NaN)
        for xx, yy in self.get_neighbors(x,y):
            xmask = xx - x + 1 # board to mask coordinates (x,y -> 1,1)
            ymask = yy - y + 1
            maskidx = xmask + ymask * self.bitdim # get equivalent index in 1D bitmask (centre -> 4)

            tset = self.tset[xx,yy]
            state = self.tile[xx,yy]
            if tset > NULLIDX:
                bm = self.tilesets[tset].bitmasks[state]
                bitmask[xmask,ymask] = bm[self.nbits -1 - maskidx] # get opposite bit (NW neighbor -> SE bit)
        bitmask = bitmask.reshape((bitmask.size,)) # convert to 1D
        return bitmask
    

    def autotile(self,  x : int, y : int, tset : int) -> None:
        '''
        Place a tile and update the states of it and its neighbors
        '''
        # check location in bounds
        if not self.check_in_bounds(x,y) : return

        # set current tile
        self._apply_autotile(x,y,tset)

        # update neighbors
        for xx, yy in self.get_neighbors(x,y):
            if self.mask[xx,yy] > 0 : self._apply_autotile(xx,yy,self.tset[xx,yy])

    def _apply_autotile(self, x : int, y : int, tset : int = NULLIDX) -> None:
        '''
        determine set tile using autotiling
        NOTE: assumes valid coordinates
        x, y: coordinates position of tile
        tset: index of tile set (-1 -> none; remove tile)
        '''
        if tset > NULLIDX:
            tileset = self.tilesets[tset]

            # compare local mask to tile bitmasks
            bitmask = self._get_local_bitmask(x,y,tileset.mask)
            delta = np.abs(tileset.bitmasks - bitmask)
            delta = np.sum(delta * self.weights, axis=1)

            # get closest-matching tile
            minidx = np.argmin(delta) # first minimum
            self._set_tile(x,y,tset,minidx)

            # check for mismatch behaviour
            '''
            minidx = np.where(delta == delta.min())[0] # list of all minima
            if len(minidx) > 1: # report imperfect matches
                print(f'Imperfect match! Candidates: {bitmask}')
                for i in minidx : print(f'   {tileset.bitmasks[i,:]}')
            '''
        
        else:
            self._remove_tile(x,y)

    
    def _get_local_bitmask(self, x : int, y : int, mask : int = NULLIDX) -> np.ndarray:
        '''
        Get the 1D bitmask centered on the given point
        x, y: coordinates
        mask: mask to filter for (-1 -> no filter)
        '''
        xboard, yboard =  self.get_neighbors(x,y).T
        xmask = xboard - x + 1 # central tile -> [1,1]
        ymask = yboard - y + 1

        bitmask = np.zeros((self.bitdim,self.bitdim),dtype=int)
        bitmask[xmask,ymask] = self.mask[xboard,yboard] # copy masks for valid tile position from board
        bitmask = bitmask.reshape((self.nbits,)) # convert to 1D

        if mask > NULLIDX : bitmask[bitmask != mask] = 0 # treat tiles with different mask as empty
        return bitmask


    def check_in_bounds(self, x : int, y : int) -> bool:
        """
        Check valid tile coordinates
        """
        if x < 0 or x >= self.dim[0] or y < 0 or y >= self.dim[1]:
            print('Index ('+str(x)+','+str(y)+') out of bounds!')
            return False
        return True
    
    def print_states(self) -> None:
         print(np.sum(self.states,2))



    def get_neighbors(self, x : int, y : int, inclusive : bool = False) -> np.ndarray:
        '''
        get coordinates of valid tiles neighboring tile at given position
        Returns (n x 2) array of x, y coordinates
        Inclusive: return reference / central tile
        '''
        x0 = x-1 if x > 0 else x
        x1 = x+2 if x < self.dim[0] -1 else x+1
        y0 = y-1 if y > 0 else y
        y1 = y+2 if y < self.dim[1] -1 else y+1
        coords = [(xx,yy) for xx in range(x0,x1) for yy in range(y0,y1) if (xx != x or yy != y) or inclusive]
        return np.array(coords)