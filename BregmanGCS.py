''''
Author: Jochem Mullink
Title: BregmanGCS
Date: 29-4-2020

Description: An implementation of the 
global convergence segmentation algorithm from 
the paper "Geometric Applications of the split
bregman method: Segmentation and surface 
reconstruction", Goldstein et al, 2009.
This minimization problem is based on the 
Mumford-shah model and is implemented using
Bregman iterations.

TODO:
- Add comments to the code.
- Implement a termination algorithm based on the
    energy of the solution.
'''

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage import color, io

def grad(u):        
    return np.gradient(u)

def div(p1,p2):
    """Calculate the divergence between p1 and p2
    """
    shape = p1.shape
    diff0 = np.diff(np.concatenate((np.zeros((1,shape[1])),p1),axis=0),axis=0)
    diff1 = np.diff(np.concatenate((np.zeros((shape[0],1)),p2),axis=1),axis=1)  
    return diff0 + diff1

def GS_u(u, r, d, b, mu, lambda1):
    b1 = b[0]
    b2 = b[1]
    d1 = d[0]
    d2 = d[1]
    
    averaging = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])/4
    
    alpha = div(b1 - d1, b2 - d2)
    beta = convolve2d(u, averaging, mode='same')  - (mu/lambda1 * r + alpha)/4
    u = np.maximum(np.minimum(beta,1),0)    
    return u

def shrink(gu, b, lambda1, g):
    gx = gu[0]
    gy = gu[1]
    bx = b[0]
    by = b[1]
    bx = bx + gx
    by = by + gy
    bnorm = np.maximum((bx**2 + by**2)**(1/2),1e-8)
    bx = np.maximum(bx - lambda1/g, 0) * bx / bnorm
    by = np.maximum(by - lambda1/g, 0) * by / bnorm        
    return [bx, by]

def H(image, c1, c2, mu, gu):
    return np.sum( mu*(c1-image)**2 - mu*(c2 - image)**2 + np.sqrt(gu[0]**2+gu[1]**2) )

def BregmanGCS(image, mu = 1., lambda1 = 1., beta = 1., max_iter=10):
    
    omega = image > mu
    c1 = np.mean( image[omega] )
    c2 = np.mean( image[np.logical_not(omega)] )
    
    b = [np.zeros(image.shape), np.zeros(image.shape)]
    
    u = image.copy()
    d = grad(u)
    
    if beta == None:
        g = np.ones(image.shape)
    else:
        g = 1/(1+(d[0]**2 + d[1]**2))
    
    energies = []
    energies.append(H(image,c1, c2, mu, d))
    
    for _ in range(max_iter):
        
        
        r = (c1 - image)**2 - (c2 - image)**2
        
        for _ in range(3):        
            u = GS_u(u, r, d, b, mu, lambda1)
        
        gu = grad(u)
        
        d = shrink(gu, b, lambda1, g)
        
        bx = b[0] + gu[0] - d[0]
        by = b[1] + gu[1] - d[1]
        b = [bx,by]
                
        omega = u > mu
        
        c1 = np.mean( image[omega] )
        c2 = np.mean( image[np.logical_not(omega)] )
        
        energies.append(H(image,c1, c2, mu, gu))
        
    print(energies)
    return u, d, b, c1, c2

def normalize(image):
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    return image

if __name__ == '__main__':
      
    # Suggested parameters for intensity_circle:
    mu = 0.2 # A suitable value for mu depends on the amount of noise.
    lambda1 = 0.5
    beta = 5
    n_iterations = 4
    img = io.imread('intensity_circle.png')
    
    # Suggested parameters for intensity_circle
    # mu = 0.5 # A suitable value for mu depends on the amount of noise.
    # lambda1 = 0.5
    # beta = 5
    # n_iterations = 4
    # img = io.imread('CT38_13.jpg')
    
    image = color.rgb2gray(img)
    image = normalize(image)
    
    sigma = 0.05
    image_wn = image + np.random.normal(0, sigma, image.shape)

    image_wn = normalize(image_wn)
        
    u, d, b, c1, c2 = BregmanGCS(image_wn, 
                                 mu, 
                                 lambda1,  
                                 beta = beta, 
                                 max_iter=n_iterations)
    
    segmentation = u > mu
        
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
    title = "u - {} iterations".format(n_iterations)
    ax[2].set_title(title, fontsize=12)

    ax[3].imshow(segmentation, cmap="gray")
    ax[3].set_axis_off()
    ax[3].set_title("segmentation", fontsize=12)
    
    fig.tight_layout()
    plt.show()
    
    
    