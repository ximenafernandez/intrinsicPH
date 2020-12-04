from fermat import Fermat
from scipy.spatial import  distance_matrix
from ripser import Rips
import matplotlib.pyplot as plt


def compute_fermat_distance(data, p):
    
    #Compute euclidean distances
    distances = distance_matrix(data,data)
    
    # Initialize the model
    fermat = Fermat(alpha = p, path_method='FW') 

    # Fit
    fermat.fit(distances)
    
    ##Compute Fermat distances
    fermat_dist = fermat.get_distances()
    
    return  fermat_dist


def compute_adj_distance(data, k):
    
    distances = distance_matrix(data,data)

    # Initialize the model
    f_aprox_D = Fermat(1, path_method='D', k=k) 

    # Fit
    f_aprox_D.fit(distances)
    adj_dist = f_aprox_D.get_distances() 
    
    return adj_dist

def Fermat_dgm(data, p, rescaled = False, d=None, mu=None):
    distance_matrix = compute_fermat_distance(data, p)
    if rescaled:
        distance_matrix = (distance_matrix*len(data)**((p-1)/d))/mu
    rips = Rips()
    dgms = rips.fit_transform(distance_matrix, distance_matrix=True)
    fig = plt.figure()
    rips.plot(dgms, lifetime = True)
    plt.title('Fermat distance with p = %s'%(p))
    return dgms