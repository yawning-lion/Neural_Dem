import jax.numpy as np

i32 = np.int32
f64 = np.float64
from matplotlib import pyplot as plt

with open('data/V.npy', 'rb') as f:
	v = np.load(f)*1*(2/3)

# with open('data/V_list.npy', 'rb') as f:
#     V = np.load(f)
with open('data/test_z.npy', 'rb') as f:
    z = np.load(f)
T = np.power(z,2)
T = np.sum(T,axis = 1).reshape(-1)*0.5
t = np.arange(100000)

plt.plot(t,T/T[0])
plt.show()
print(T[19999]/T[0])
plt.plot(t,v/T[0])
plt.show()

plt.plot(t,(T+v)/T[0])
plt.show()
