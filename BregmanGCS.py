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
import skimage
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage import color, io
from skimage import data
from skimage.exposure import rescale_intensity
from skimage.segmentation import chan_vese

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
    mu = 0.55 # A suitable value for mu depends on the amount of noise.
    lambda1 = 0.6
    beta = 5
    n_iterations = 4

    #img = io.imread('test images/intensity_circle.png')
    img = io.imread('CT38/images/CT38_1.jpg')
    #img = io.imread('test images/blurrycircles.png')
    image = color.rgb2gray(img)
    image = normalize(image)

    u, d, b, c1, c2 = BregmanGCS(image,
                                 mu=mu,
                                 lambda1=lambda1,
                                 beta=beta,
                                 max_iter=n_iterations)

    segmentation = u > mu

    cv1 = chan_vese(image, mu=0.15, lambda1=1, lambda2=1, tol=1e-3, max_iter=50, extended_output=True)

    fig, axes = plt.subplots(1, 3)
    ax = axes.flatten()

    ax[0].imshow(img, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original image", fontsize=12)

    ax[1].imshow(cv1[0], cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title("Chan-Vese")

    ax[2].imshow(segmentation, cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Split Bregman")

    plt.show()





    """
    

    u, d, b, c1, c2 = BregmanGCS(image,
                                 mu=mu,
                                 lambda1=lambda1,
                                 beta=beta,
                                 max_iter=n_iterations)

    segmentation = u > mu

    fig, axes = plt.subplots(1,2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original image", fontsize=11)

    ax[1].imshow(segmentation, cmap="gray")
    ax[0].set_axis_off()
    plt.show()
    """

    """
    # test effect of noise
    img = skimage.img_as_float(data.binary_blobs(length=128, seed=1))
    img = normalize(img)

    sigma = 0.25
    data1 = img + np.random.normal(loc=0, scale=sigma, size=img.shape)
    data1 = rescale_intensity(data1, in_range=(-sigma, 1+sigma), out_range=(-1, 1))
    data1 = normalize(data1)

    sigma = 0.5
    data2 = img + np.random.normal(loc=0, scale=sigma, size=img.shape)
    data2 = rescale_intensity(data2, in_range=(-sigma, 1 + sigma), out_range=(-1, 1))
    data2 = normalize(data2)

    sigma = 0.75
    data3 = img + np.random.normal(loc=0, scale=sigma, size=img.shape)
    data3 = rescale_intensity(data3, in_range=(-sigma, 1 + sigma), out_range=(-1, 1))
    data3 = normalize(data3)

    sigma = 1
    data4 = img + np.random.normal(loc=0, scale=sigma, size=img.shape)
    data4 = rescale_intensity(data4, in_range=(-sigma, 1 + sigma), out_range=(-1, 1))
    data4 = normalize(data4)


    u, d, b, c1, c2 = BregmanGCS(data1,
                                 mu=mu,
                                 lambda1=lambda1,
                                 beta = beta, 
                                 max_iter=n_iterations)
    
    segmentation1 = u > mu
    d1_bregman = img - segmentation1

    u, d, b, c1, c2 = BregmanGCS(data2,
                                 mu=mu,
                                 lambda1=lambda1,
                                 beta = beta,
                                 max_iter=n_iterations)
    segmentation2 = u > mu
    d2_bregman = img - segmentation2

    u, d, b, c1, c2 = BregmanGCS(data3,
                                 mu=mu,
                                 lambda1=lambda1,
                                 beta=beta,
                                 max_iter=n_iterations)
    segmentation3 = u > mu
    d3_bregman = img - segmentation3

    u, d, b, c1, c2 = BregmanGCS(data4,
                                 mu=mu,
                                 lambda1=lambda1,
                                 beta=beta,
                                 max_iter=n_iterations)
    segmentation4 = u > mu
    d4_bregman = img - segmentation4
    


    # chan-vese segmentation
    cv1 = chan_vese(data1, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=50, extended_output=True)
    d1 = img - cv1[0]
    cv2 = chan_vese(data2, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=50, extended_output=True)
    d2 = img - cv2[0]
    cv3 = chan_vese(data3, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=50, extended_output=True)
    d3 = img - cv3[0]
    cv4 = chan_vese(data4, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=50, extended_output=True)
    d4 = img - cv4[0]
    """
    """
    # plot effect of noise
    fig, axes = plt.subplots(4, 5, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(data1, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Noise level $\sigma = 0.25$", fontsize=11)

    ax[1].imshow(cv1[0], cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title("Chan-Vese", fontsize=11)

    ax[2].imshow(segmentation1, cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Split Bregman", fontsize=11)

    ax[3].imshow(d1, cmap="gray")
    ax[3].set_axis_off()
    ax[3].set_title("Chan-Vese error", fontsize=11)

    ax[4].imshow(d1_bregman, cmap="gray")
    ax[4].set_axis_off()
    ax[4].set_title("Split Bregman error", fontsize=11)

    ax[5].imshow(data2, cmap="gray")
    ax[5].set_axis_off()
    ax[5].set_title("Noise level $\sigma = 0.5$", fontsize=11)

    ax[6].imshow(cv2[0], cmap="gray")
    ax[6].set_axis_off()

    ax[7].imshow(segmentation2, cmap="gray")
    ax[7].set_axis_off()

    ax[8].imshow(d2, cmap="gray")
    ax[8].set_axis_off()

    ax[9].imshow(d2_bregman, cmap="gray")
    ax[9].set_axis_off()

    ax[10].imshow(data3, cmap="gray")
    ax[10].set_axis_off()
    ax[10].set_title("Noise level $\sigma = 0.75$", fontsize=11)

    ax[11].imshow(cv3[0], cmap="gray")
    ax[11].set_axis_off()

    ax[12].imshow(segmentation3, cmap="gray")
    ax[12].set_axis_off()

    ax[13].imshow(d3, cmap="gray")
    ax[13].set_axis_off()

    ax[14].imshow(d3_bregman, cmap="gray")
    ax[14].set_axis_off()

    ax[15].imshow(data4, cmap="gray")
    ax[15].set_axis_off()
    ax[15].set_title("Noise level $\sigma = 1$", fontsize=11)

    ax[16].imshow(cv4[0], cmap="gray")
    ax[16].set_axis_off()

    ax[17].imshow(segmentation4, cmap="gray")
    ax[17].set_axis_off()

    ax[18].imshow(d4, cmap="gray")
    ax[18].set_axis_off()

    ax[19].imshow(d4_bregman, cmap="gray")
    ax[19].set_axis_off()


    fig.tight_layout()
    plt.show()
    """




    #plt.imshow(segmentation, cmap="gray")
    #plt.imsave("segmentation43.jpg", segmentation)
    
    