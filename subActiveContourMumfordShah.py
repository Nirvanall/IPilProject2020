import numpy as np
from subTotalVariation import _total_variation

def r1(image, c1, c2):
    return (image-c1)**2 - (image-c2)**2

def nu(v):
    return np.maximum(0, 2*np.abs(v-0.5)-1)


def _minimization2(u, c1, c2, image,mask):
    temp = u - theta1 * lambda1 *r1(image, c1, c2)
    temp = np.maximum(temp,0.0)
    temp = np.minimum(temp, 1.0)
    return temp


def ActiveContourMS( image,
                     lambda1=0.1,
                     theta1=1.0, 
                     epsilon=0.1,
                     maxit =100,
                     mask=None
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
        
    if mask == None:
        g = np.ones(image.shape,dtype=bool)
        
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
        
    i = 0
    
    new_u = np.zeros(shape)
    new_v = np.zeros(shape)
    difference = epsilon + 1.0
    
    # mu = np.mean(image)
    mu = 0.5
    c1 = np.mean(image[image>mu])
    c2 = np.mean(image[image<=mu])
    
    while (i < maxit) & (difference > epsilon):
        # if i % 30 == 29:
        #     mu = np.mean(new_u)
        #     c1 = np.mean([new_u>mu])
        #     c2 = np.mean(new_u[new_u<=mu])
        #     print(c1,c2)
        i += 1
        
        old_u = new_u
        old_v = new_v
        
        # mu = np.mean(new_u)
        # c1 = np.mean(new_u[new_u>mu])
        # c2 = np.mean(new_u[new_u<=mu])
        # c1 = 0.7
        # c2 = 0.05
        
        new_u, _ = _total_variation(new_v,theta1,maxit=60)
        new_v = _minimization2(new_u, c1, c2, image, mask)
               
        
        difference = max([np.linalg.norm(new_v-old_v,ord=1),np.linalg.norm(new_u-old_u,ord=1)])
    
    return new_u, new_v

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
    
   
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import color, io


    # img = io.imread('intensity_circle.png')
    img = io.imread('CT38_13.jpg')
    
    img = color.rgb2gray(img)
    image = img - np.mean(img)

    
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    
    
    # Feel free to play around with the parameters to see how they impact the result
    n_iterations = 10
    lambda1 = 0.1
    theta1 = 1.
    
    u, v = ActiveContourMS(image,
                            lambda1=lambda1, 
                            theta1=theta1, maxit=n_iterations)
    partition = u >= np.mean(u)
    
    
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(u, cmap="gray")
    ax[1].set_axis_off()
    title = "u - {} iterations".format(n_iterations)
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(v, cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("v", fontsize=12)
    
    ax[3].imshow(partition, cmap="gray")
    ax[3].set_axis_off()
    ax[3].set_title("partition", fontsize=12)
    

    fig.tight_layout()
    plt.show()
    
    