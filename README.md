# WFC
Basic, bitmask-based demos of wave function collapse and autotiling

wfc_demo.py: a simple wave function collapse algorithm, where tiles are placed sequentially and the state of a newly-placed tile is restricted by those of its neighbors.
A bitmask dictates tiling rules.

autotile.py: a demonstration of autotiling; tiles are placed randomly, and a newly-placed tile will update the states of its neighbors.
Empty spaces are considered non-matches to restrict valid states.
This algorithm accommodates multiple tilesets and bitmasks, and different tilesets can be set to tile together.

tile.py: library for creating and manipulating tiles.
A second library for boards and tile-setting rules will hopefully be added in the future.

Tilesets created by Cup Nooble on ich.io: https://cupnooble.itch.io/
