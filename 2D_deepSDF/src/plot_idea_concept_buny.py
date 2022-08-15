from vedo import dataurl, Mesh, Arrows, show

s1 = Mesh(dataurl + "bunny.obj").c("gold").smooth()
s1.smooth().computeNormals()
s1.c('light blue').lw(0).lighting('glossy').phong()
s2 = s1.clone(deep=False).mirror("y")
s2.pos(0,0.5,0).c("light green").flag('mirrored')
s3 = s1.clone(deep=False).mirror("x")
s3.pos(0.2,0.15,0).c("gold").flag('mirrored')
s4 = s1.clone(deep=False).mirror("z")
s4.pos(-0.1,0.5,0.1).c("red").flag('mirrored')
s5 = s1.clone(deep=False).mirror("z")
s5.pos(-0.15,0.15,-0.1).c("purple").flag('mirrored')
show(s1,s2,s3,s4,s5)