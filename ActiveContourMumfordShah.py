"""
Author: Jochem Mullink
Date: 1/5/2020
Title: ActiveContourMumfordShah

Description: This is an implementation of a segmentation
algorithm based on the mumford shah model. This algorithm
is described in the paper "Fast global minimization of
the active contour/snake model", Bresson et al, 2005.
Essentially this is a convex relaxation of the
mumford shah model.
"""

import numpy as np

def gradient(x):
    """Calculate gradient from x"""
    return np.gradient(x)

def div(p1,p2):
    """Calculate the divergence between p1 and p2
    """
    shape = p1.shape
    diff0 = np.diff(np.concatenate((np.zeros((1,shape[1])),p1),axis=0),axis=0)
    diff1 = np.diff(np.concatenate((np.zeros((shape[0],1)),p2),axis=1),axis=1)  
    return diff0 + diff1

def iteration(p1,p2,dt,v,theta1,g):
    """Calculate next iteration of total_variation"""
    divp = div(p1,p2)
    gradp = gradient(divp - v/theta1)
    return (p1 + dt*gradp[0])/(1 + dt/g*np.abs(gradp[0])),(p2 + dt*gradp[1])/(1 + dt/g*np.abs(gradp[1]))

def _total_variation(v, theta1, g, dt=1/16, maxit=10, tol=1.):
    """Solves minimization problem 1 from page 5.
    min_u TV(u) + ||u-v||_2
    """
    shape = v.shape
    p1 = np.zeros(shape)
    p2 = np.zeros(shape)
    
    p1_old = p1
    p2_old = p2
    i  = 0
    eps = tol + 1.
    
    while (i < maxit) & (tol < eps):
        p1, p2 = iteration(p1,p2,dt,v,theta1,g)
        
        i += 1
        eps = np.linalg.norm(p1-p1_old) + np.linalg.norm(p2-p2_old)
        p1_old = p1
        p2_old = p2        
    return v - theta1 * div(p1,p2), i

def r1(image, c1, c2):
    return (image-c1)**2 - (image-c2)**2

def nu(v):
    return np.maximum(0, 2*np.abs(v-0.5)-1)


def _minimization2(u, c1, c2, image):
    temp = u - theta1 * lambda1 *r1(image, c1, c2)
    temp = np.maximum(temp,0.0)
    temp = np.minimum(temp, 1.0)
    return temp


def ActiveContourMS( image,
                     lambda1=0.1,
                     theta1=1.0, 
                     epsilon=0.1,
                     maxit =100,
                     beta=None
                     ):
    """
    

    Parameters
    ----------
    image : np.array
        Gray values of input image
    lambda1 : float, optional
        A regularization parameter that limits the L1-norm of v. The default is 0.1.
    theta1 : float, optional
        A regularization parameter that decreases the effect of the data fidelity term.
        The default is 1.0.
    epsilon : float, optional
        Convergence criterion. The default is 0.1.
    maxit : int, optional
        Maximum number of iterations. The default is 100.

    Returns
    -------
    u,v

    """
    
    shape = image.shape
    
    if len(shape) != 2:
        raise ValueError("Input image should be a 2D array.")
        
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
        
    i = 0
    
    new_u = image.copy()
    new_v = image.copy()
    difference = epsilon + 1.0
    
    c1 = np.mean(image[image>lambda1])
    c2 = np.mean(image[image<=lambda1])
    
    gimage = gradient(image)
    
    if beta == None:
        g = np.ones(image.shape)
    else:
        g = 1/(1+(gimage[0]**2 + gimage[1]**2))
    
    while (i < maxit) & (difference > epsilon):
        i += 1
        
        old_u = new_u
        old_v = new_v
        
        new_u, _ = _total_variation(new_v,theta1,g,maxit=10)
        new_v = _minimization2(new_u, c1, c2, image)
        
        
        omega = new_u>lambda1
        c1 = np.mean(image[omega])
        c2 = np.mean(image[np.logical_not(omega)])
        
        difference = max([np.linalg.norm(new_v-old_v,ord=1),np.linalg.norm(new_u-old_u,ord=1)])
    
    return new_u, new_v, omega
    
def normalize(image):
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    return image
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import color, io


    # img = io.imread('intensity_circle.png')
    img = io.imread('CT38_13.jpg')
    
    image = color.rgb2gray(img)
    image = normalize(image)
    
    sigma = 0.05
    image_wn = image + np.random.normal(0, sigma, image.shape)

    image_wn = normalize(image_wn)
    
    
    # Feel free to play around with the parameters to see how they impact the result
    n_iterations = 60
    lambda1 = 0.4
    theta1 = 1.
    
    u, v, partition = ActiveContourMS(image_wn,
                                      lambda1=lambda1, 
                                      theta1=theta1, 
                                      maxit=n_iterations,
                                      beta=1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)
    
    ax[1].imshow(image_wn, cmap="gray")
    ax[1].set_axis_off()
    title = "Original image with noise added"
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(u, cmap="gray")
    ax[2].set_axis_off()
    title = "u - {} iterations of the Active Contour method".format(n_iterations)
    ax[2].set_title(title, fontsize=12)

    ax[3].imshow(partition, cmap="gray")
    ax[3].set_axis_off()
    ax[3].set_title("segmentation", fontsize=12)
    
    fig.tight_layout()
    plt.show()
    
    