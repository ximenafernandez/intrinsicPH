import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
from   scipy.integrate import odeint
import numpy as np

def Lorenz(X, t, sigma, beta, rho):
    '''
    The Lorenz equations.
    '''
    x, y, z = X
    dx = -sigma * (x - y)
    dy = rho*x - y - x*z
    dz = -beta*z + x*y
    return (dx, dy, dz)


def Rossler(X, t, a, b, c):
    '''
    The Rossler equations.
    '''
    x, y, z = X
    x_dot = -y - z
    y_dot = x + a*y
    z_dot = b + x*z - c*z
    return (x_dot, y_dot, z_dot)

def simulate(equations, a, b, c, x0, y0, z0, tmax, n):
    '''
    Integrate equations on the time grid t.
    '''
    
    t = np.linspace(0, tmax, n)
    f = odeint(equations, (x0, y0, z0), t, args=(a, b, c))
    x, y, z = f.T
    return x,y,z

def plot_trajectories(x,y,z,title, type_plot):
    '''
    'Plot 3 trajectories using a Matplotlib 3D projection.
    '''
    
    fig = plt.figure(figsize=(8,6))

    ax = Axes3D(fig)
    
    if type_plot == 'line':
        ax.plot(x, y, z, 'b-', lw=0.5, color='lightcoral')
    
    if type_plot == 'scatter':
        ax.scatter(x, y, z, s=0.5)

    plt.tick_params(labelsize=10)
    ax.set_title(title, fontsize=15)

    plt.show()
    

def voxel_down_sample(data, voxel_size):
    '''
    Down sample the point cloud according to a grid of fixed size.
    INPUT: 
    - data: point cloud, subset of a R^n
    - voxel_size: float number indicating the size of the cubical grid of R^n
    OUTPUT: new point cloud obtained from data by computing the mean of the points of the original point cloud points that are inside each grid cube.
    '''
    
    dim = len(data[0])
    m = {} #min of the data in each dimension
    for i in range(dim):
        m[i] = data[:,i].min()
    grid_data = {}
    for x in data:
        pos = {}
        for i in range(dim):
          pos[i] = int((x[i]-m[i])/voxel_size)
        tuple_pos = tuple(pos.values())
        if tuple_pos not in grid_data.keys():
          grid_data[tuple_pos] = np.array([x])
        else:
          grid_data[tuple_pos] = np.append(grid_data[tuple_pos], [x], axis=0)
    mean_grid_data = dict(map(lambda t: (t[0], t[1].mean(axis = 0)), grid_data.items()))
    return np.array(list(mean_grid_data.values()))

def delay_embedding(s, T, d, step=1):
    '''
    Delay embedding of a time series
    
    INPUT:
    - s: 1-dimensional array, the time series.
    - T: an integer, the delay
    - d: an integer, the ambient dimension of the embedding
    - step: an integer, the step used to read the time series
    
    OUTPUT: an array, representing points in R^d
    '''
    
    N = len(s)
    X = []
    for i in range(d):
        X.append(s[i*T: N-(d-1-i)*T:step])
    return X

def moving_average(x, w):
    '''
    Moving average of a curve x with window w.
    '''
    return np.convolve(x, np.ones(w), 'valid') / w