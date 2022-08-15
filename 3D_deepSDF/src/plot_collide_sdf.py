from vedo import dataurl, Mesh, Arrows, show
import jax.numpy as np

batch_verts = np.load('data/temp_data/batch_verts.npy')
faces = np.load('data/temp_data/faces.npy')

verts0 = batch_verts[0]
verts1 = batch_verts[1]
verts2 = batch_verts[2]


mesh0 = Mesh([verts0, faces])
mesh1 = Mesh([verts1, faces])
mesh2 = Mesh([verts2, faces])



mesh1.smooth().computeNormals()
mesh1.lw(0).phong()
mesh1.pos(3,0,0)

mesh2.smooth().computeNormals()
mesh2.c('grey4', alpha = 0.5).lw(0).phong()
mesh2.pos(2.1,2.0,0)

mesh1.distanceTo(mesh2, signed=True)
mesh1.cmap('RdBu').addScalarBar('Signed\nDistance')

#show(mesh0, mesh1, mesh2, mesh3, mesh4, mesh5,mesh6, __doc__, axes=2, viewup="z", bg2='ly').close()
show(mesh1, mesh2)
