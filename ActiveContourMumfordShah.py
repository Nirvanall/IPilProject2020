import numpy as np
from TotalVariation import _total_variation

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
                     maxit =100
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
    
    while (i < maxit) & (difference > epsilon):
        i += 1
        
        old_u = new_u
        old_v = new_v
        
        new_u, _ = _total_variation(new_v,theta1,maxit=10)
        new_v = _minimization2(new_u, c1, c2, image)
        
        
        omega = new_u>lambda1
        c1 = np.mean(image[omega])
        c2 = np.mean(image[np.logical_not(omega)])
        
        difference = max([np.linalg.norm(new_v-old_v,ord=1),np.linalg.norm(new_u-old_u,ord=1)])
    
    return new_u, new_v, omega

def _k_means(u, k):
    n = u.size
    shape = u.shape
    X = np.zeros((n,1))
    i = 0
    
    for index in np.ndindex(shape):
        X[i,0] = u[index]
        i+=1
    
    from sklearn.cluster import k_means
    kmeans = k_means(X, k)
    labels = kmeans[1]
    labels = labels.reshape(shape)
    return labels
    
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
                                      theta1=theta1, maxit=n_iterations)
    
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
    
    