from lib.coarsening import coarsen
from lib.grid_graph import radial_graph
# from lib.coarsening import perm_data


# A = radial_graph()
A = radial_graph(t_units=2, number_edges=2)
coarsening_levels = 2
L, perm = coarsen(A, coarsening_levels)
