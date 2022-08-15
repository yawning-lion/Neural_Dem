from vedo import dataurl, Mesh, Arrows, show
import jax.numpy as np

batch_verts = np.load('data/temp_data/batch_verts.npy')
faces = np.load('data/temp_data/faces.npy')

verts0 = batch_verts[0]
verts1 = batch_verts[1]
verts2 = batch_verts[2]
verts3 = batch_verts[3]
verts4 = batch_verts[4]
verts5 = batch_verts[5]
verts6 = batch_verts[6]


mesh0 = Mesh([verts0, faces])
mesh1 = Mesh([verts1, faces])
mesh2 = Mesh([verts2, faces])
mesh3 = Mesh([verts3, faces])
mesh4 = Mesh([verts4, faces])
mesh5 = Mesh([verts5, faces])
mesh6 = Mesh([verts6, faces])

mesh0.smooth().computeNormals()
mesh0.c('light blue').lw(0).phong().flag('mesh0')
mesh0.pos(0,0,0)

mesh1.smooth().computeNormals()
mesh1.c('light blue').lw(0).phong()
mesh1.pos(3,0,0)

mesh2.smooth().computeNormals()
mesh2.c('light blue').lw(0).phong()
mesh2.pos(2.1,2.0,0)

mesh3.smooth().computeNormals()
mesh3.c('light blue').lw(0).phong()
mesh3.pos(1.2,-2.5,0)

mesh4.smooth().computeNormals()
mesh4.c('light blue').lw(0).phong()
mesh4.pos(-1.5,-1.5,0)

mesh5.smooth().computeNormals()
mesh5.c('light blue').lw(0).phong()
mesh5.pos(-2.0,1.0,0)

mesh6.smooth().computeNormals()
mesh6.c('light blue').lw(0).phong()
mesh6.pos(-0.7,2.5,0)


#show(mesh0, mesh1, mesh2, mesh3, mesh4, mesh5,mesh6, __doc__, axes=2, viewup="z", bg2='ly').close()
show(mesh0, mesh1, mesh2, mesh3, mesh4, mesh5,mesh6)
