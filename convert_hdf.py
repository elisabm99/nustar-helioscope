from pyhdf.SD import SD, SDC
import numpy as np


FILE_NAME = 'br002.hdf'
hdf = SD(FILE_NAME, SDC.READ)
print(hdf.datasets())
phi = hdf.select('fakeDim0')[:]
theta = hdf.select('fakeDim1')[:]
rs = hdf.select('fakeDim2')[:]
b = hdf.select('Data-Set-2')[:]
np.savez('br2019.npz', b=b, r=rs, theta=theta, phi=phi)


FILE_NAME = 'bt002.hdf'
hdf = SD(FILE_NAME, SDC.READ)
print(hdf.datasets())
phi = hdf.select('fakeDim0')[:]
theta = hdf.select('fakeDim1')[:]
rs = hdf.select('fakeDim2')[:]
b = hdf.select('Data-Set-2')[:]
np.savez('bt2019.npz', b=b, r=rs, theta=theta, phi=phi)


FILE_NAME = 'bp002.hdf'
hdf = SD(FILE_NAME, SDC.READ)
print(hdf.datasets())
phi = hdf.select('fakeDim0')[:]
theta = hdf.select('fakeDim1')[:]
rs = hdf.select('fakeDim2')[:]
b = hdf.select('Data-Set-2')[:]
np.savez('bp2019.npz', b=b, r=rs, theta=theta, phi=phi)


