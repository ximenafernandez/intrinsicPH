import random
import numpy as np
import matplotlib.colors as colors
from   numpy import linspace,pi,cos,sin

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#Eyeglasses

def generate_noise(type_noise, var, n):
    if type_noise == 'normal':
        return np.random.normal(0, var, n)
    if type_noise == 'uniform':
        return np.random.uniform(-var, var, n)
    
def eyeglasses(n_obs, n_out, type_noise, var):
    '''
    Sample on eyeglasses curve with noise and outliers.
    
    Input:
    n_obs: an integer, number of points on the Eyeglasses
    n_out: an integer, number on points randomly drawn in [-3,3]x[-1.5,1.5]
    type_noise: string, 'normal' or uniform
    var: a float, variance of the noise
    seed: an integer, the seed for the random generation of the noise
    
    Output:
    data: a nx2 array, representing points in R^2 
    '''
    
    n = int(0.85*(n_obs/2))
    m = int(0.15*(n_obs/2))
    phi1 = np.linspace(np.pi-1.2, 2*np.pi+1.2, n)
    phi2 = np.linspace(np.pi+1.92, 2*np.pi+4.35, n)
    seg  = np.linspace(-0.53, 0.53, m)
    
    if type_noise == 'normal':
        noise_1 = np.random.normal(0, var, n)
        noise_2 = np.random.normal(0, var, m)
    if type_noise == 'uniform':
        noise_1 = np.random.uniform(-var, var, n)
        noise_2 = np.random.uniform(-var, var, m)
        
    x1 = np.sin(phi1)+ generate_noise(type_noise, var, n) - 1.5
    y1 = np.cos(phi1)+ generate_noise(type_noise, var, n)
    x2 = np.sin(phi2)+ generate_noise(type_noise, var, n) + 1.5
    y2 = np.cos(phi2)+ generate_noise(type_noise, var, n)
    x3 = seg   + generate_noise(type_noise, var, m)
    y3 = 0.35  + generate_noise(type_noise, var, m)
    x4 = seg   + generate_noise(type_noise, var, m)
    y4 = -0.35 + generate_noise(type_noise, var, m)
    
    X_obs = np.concatenate([x1,x2, x3, x4])
    Y_obs = np.concatenate([y1,y2, y3, y4])
    
    X_out = (np.random.rand(n_out)-0.5)*6
    Y_out = (np.random.rand(n_out)-0.5)*3
    
    X = np.concatenate([X_obs, X_out])
    Y = np.concatenate([Y_obs, Y_out])
    
    data = np.column_stack((X,Y))
    
    return data


### Trefoil

def trefoil(n_obs, type_noise, var, seed):
    '''
    Sample on trifoil curve with noise and outliers.
    
    Input:
    n_obs: an integer, number of points on the Trifol
    type_noise: string, 'normal' or uniform
    var: a float, variance of the noise    
    Output:
    data: a nx2 array, representing points in R^2
    '''
    
    phi = linspace(0,2*pi,n_obs)
    
    X_obs = sin(phi)+2*sin(2*phi) + generate_noise(type_noise, var, n_obs)
    Y_obs = cos(phi)-2*cos(2*phi) + generate_noise(type_noise, var, n_obs)
    Z_obs = -sin(3*phi)           + generate_noise(type_noise, var, n_obs)
    
    data = np.column_stack((X_obs,Y_obs,Z_obs))
    
    return data
    