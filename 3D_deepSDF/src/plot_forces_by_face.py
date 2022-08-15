from vedo import *
import numpy as onp


batch_verts = onp.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/batch_verts.npy")
faces = onp.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/faces.npy")

verts_1 = batch_verts[0]
faces_1 = faces
pt = [1, 0.5, 1]
R = 3.55
mesh_1 = Mesh([verts_1, faces_1])

ids = mesh_1.closestPoint(pt, radius=R, returnPointId=True)


mesh_1.deleteCellsByPointIndex(ids)
mesh_1.clean()
print(mesh_1.NPoints)

show(mesh_1)