# WFC
Basic, bitmask-based demos of wave function collapse and autotiling

wfc_demo.py: a simple wave function collapse algorithm, where tiles are placed sequentially and the state of a newly-placed tile is restricted by those of its neighbors.
A bitmask dictates tiling rules.

wfc_new.py: an updated version of wfc_demo.py using the same Board as autotile.py and making better use of vector operations. Slightly different behaviour

autotile.py: a demonstration of autotiling; tiles are placed randomly, and a newly-placed tile will update the states of its neighbors.
Empty spaces are considered non-matches to restrict valid states.
This algorithm accommodates multiple tilesets and bitmasks, and different tilesets can be set to tile together.

tile.py: for creating and manipulating tilesets.

board.py: for creating and updating boards of tiles using either autotiling or wfc rules

Tilesets created by Cup Nooble on ich.io: https://cupnooble.itch.io/
