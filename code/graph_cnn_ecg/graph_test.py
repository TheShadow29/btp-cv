from lib.coarsening import coarsen
from lib.grid_graph import radial_graph
# from lib.coarsening import perm_data


A = radial_graph()
coarsening_levels = 10
L, perm = coarsen(A, coarsening_levels)
