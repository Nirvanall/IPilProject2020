import numpy as np
from TotalVariation import _total_variation

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

def iteration(p1,p2,dt,image,v,theta1):
    """Calculate next iteration of total_variation"""
    diff = (image - v)/theta1
    divp = div(p1,p2)
    gradp = gradient(divp - diff)
    return (p1 + dt*gradp[0])/(1 + dt*gradp[0]),(p2 + dt*gradp[1])/(1 + dt*gradp[1])

def _total_variation(image, v, theta1, dt=1/16, p1=None,p2=None,maxit=10):
    """Solves minimization problem 1 from page 5.
    TV(u) + 1/(2*theta)*||u-v||^2
    """
    shape = image.shape
    if p1 == None:
        p1 = np.zeros(shape)
    if p2 == None:
        p2 = np.zeros(shape)
    
    for _ in range(maxit):
        p1, p2 = iteration(p1,p2,dt,image,v,theta1)
    return image - v - theta1 * div(p1,p2)

def _lasso(diff, c):
    """Solves minimization problem 2 from page 5.
    """
    v = np.zeros(diff.shape)
    
    mask = diff >= c
    v[mask] = diff[mask] - c
    
    mask = diff <= c
    v[mask] = diff[mask] + c
    return v

def _wavelet(diff, c):
    
    import pywt
    coeffs = pywt.wavedec2(diff, 'db1')
    
    for index, coef in enumerate(coeffs):
        if type(coef) == tuple:
            coef = list(coef)
            coef = [pywt.threshold(array,lambda1*theta1) for array in coef]
            coef = tuple(coef)
            coeffs[index] = coef
        else:
            coef = pywt.threshold(coef, c)
            coeffs[index] = coef
    return pywt.waverec2(coeffs,'db1')
                
    
            

def _tikhonov(diff, c):
    """Solves minimization problem 2 from page 5.
    """
    return diff/(1+c)



def ActiveContourROF(image,
                     filter_type = 'tikhonov',
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
    
    new_u = np.zeros(shape)
    new_v = np.zeros(shape)
    difference = epsilon + 1.0
    
    while (i < maxit) & (difference > epsilon):
        i += 1
        
        old_u = new_u
        old_v = new_v
        
        # new_u = _total_variation(image, new_v, theta1, maxit = 60)
        diff = image - new_v
        new_u, _ = _total_variation(diff,theta1,maxit=60)
        
        diff = image - new_u
        
        if filter_type == 'tikhonov':
            new_v = _tikhonov(diff, c=lambda1*theta1)
        elif filter_type == 'wavelet':
            new_v = _wavelet(diff, c=lambda1*theta1)
        elif filter_type == 'lasso':
            new_v = _lasso(diff, c=lambda1*theta1)
        
        difference = max([np.linalg.norm(new_v-old_v,ord=1),np.linalg.norm(new_u-old_u,ord=1)])
    
    return new_u, new_v

def _k_means(u, k):
    n = u.size
    shape = u.shape
    X = np.zeros((n,3))
    i = 0
    
    for index in np.ndindex(shape):
        X[i,0] = index[0]/shape[0]/10
        X[i,1] = index[1]/shape[1]/10
        X[i,2] = u[index]
        i+=1
    
    from sklearn.cluster import k_means
    kmeans = k_means(X, k)
    labels = kmeans[1]
    labels = labels.reshape(shape)
    return labels
    
   
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import color, io


    img = io.imread('intensity_circle.png')
    # img = io.imread('CT38_13.jpg')
    
    img = color.rgb2gray(img)
    image = img - np.mean(img)

    
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    
    
    # Feel free to play around with the parameters to see how they impact the result
    n_iterations = 60
    lambda1 = 0.1
    theta1 = .1
    filter_type = 'tikhonov'
    u, v = ActiveContourROF(image, filter_type = filter_type, 
                            lambda1=lambda1, 
                            theta1=theta1, maxit=n_iterations)
    
 
    
    
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
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
    
    fig.tight_layout()
    plt.show()
    
    