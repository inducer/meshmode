import numpy as np
from scipy.spatial import Delaunay

d = 4

points = np.zeros((d, 2**d))

for i in range(2**d):
    for j in range(d):
        points[j, i] = 1 if i & (1 << j) else 0


dtri = Delaunay(points.T)
print(dtri.points)
print(dtri.simplices)
