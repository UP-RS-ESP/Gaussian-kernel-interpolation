import numpy as np
from scipy.spatial import cKDTree as kdtree
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as pl

shape = (3, 4, 5) # [meters]
num = 1000
sigma = 25.0 / 2
theory = shape[0]/2

def gaussian_weights(r, sigma):
    return np.exp(-r*r/sigma/sigma/2)

# point cloud data
pts = np.c_[shape[2] * np.random.random(num),
            shape[1] * np.random.random(num),
            shape[0] * np.random.random(num)]

# interpolation variable
var = np.cos(pts[:,2]/shape[0]*np.pi)**2

fg = pl.figure(1, (8, 6))
ax = fg.add_subplot(111)

for voxel_width in np.linspace(0.05, 0.25, 5):
    print('voxel width:', voxel_width)

    # voxel bounds
    vxb = np.linspace(0, shape[2], int(shape[2]/voxel_width)+1)
    vyb = np.linspace(0, shape[1], int(shape[1]/voxel_width)+1)
    vzb = np.linspace(0, shape[0], int(shape[0]/voxel_width)+1)

    # voxel center coordinates
    vyc, vzc, vxc = np.meshgrid((vyb[1:]+vyb[:-1])/2,
                                (vzb[1:]+vzb[:-1])/2,
                                (vxb[1:]+vxb[:-1])/2)
    vxl = np.c_[vxc.ravel(), vyc.ravel(), vzc.ravel()]

    # KD-Tree
    tree = kdtree(pts)

    # it would be more honest to take a ball neighborhood,
    # but I'll just take the 20 nn for speed and simplicity
    # problem: nn might not form a sphere
    dd, ii = tree.query(vxl, k = 20)
    wi = gaussian_weights(dd, sigma)
    vi = var[ii]

    # Gaussian kernel interpolation
    var_gki = np.sum(vi*wi, axis = 1) / np.sum(wi, axis = 1)
    var_gki.shape = vzc.shape

    # digital surface model
    dsm = vzb[np.nanargmin(var_gki, axis = 0)] + voxel_width/2

    # error distribution
    bins = np.arange(-3.25, 3.50, 0.5)
    p, b = np.histogram((theory - dsm)/voxel_width, bins = bins, density = True)
    ax.plot((b[1:]+b[:-1])/2, p, label = 'voxel width %.2f' % voxel_width)

ax.set_xlabel('DSM error [voxel width]')
ax.set_ylabel('Probability')
ax.set_xlim((bins.min(), bins.max()))
pl.grid()
pl.legend()
pl.tight_layout()
pl.savefig('dsm-error-voxel-widths.png')
