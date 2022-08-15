from vedo import *
import numpy as onp

settings.useDepthPeeling = True

batch_verts = onp.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/batch_verts.npy")
faces = onp.load("/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/faces.npy")

shape_verts_0 = batch_verts[0]
shape_verts_1 = batch_verts[1]

shape_verts_0 = batch_verts[0] + [2,2,1]

mesh1 = Mesh([shape_verts_0, faces])
mesh2 = Mesh([shape_verts_1, faces])
mesh1.lw(0.1)
mesh2.c('lightgreen').bc('tomato').lw(0.1)



pt = [2, 2, 1]
R = 2.3
ids = mesh2.closestPoint(pt, radius=R, returnPointId=True)

mesh2.deleteCellsByPointIndex(ids)
mesh2.clean()

mesh1.distanceTo(mesh2, signed=True)
mesh1.cmap('hot').addScalarBar('Signed\nDistance')
show(mesh1, mesh2)