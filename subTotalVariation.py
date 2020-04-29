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

def iteration(p1,p2,dt,v,theta1):
    """Calculate next iteration of total_variation"""
    divp = div(p1,p2)
    gradp = gradient(divp - v/theta1)
    return (p1 + dt*gradp[0])/(1 + dt*gradp[0]),(p2 + dt*gradp[1])/(1 + dt*gradp[1])

def _total_variation(v, theta1, dt=1/16, maxit=10, tol=1.):
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
        p1, p2 = iteration(p1,p2,dt,v,theta1)
        
        i += 1
        eps = np.linalg.norm(p1-p1_old) + np.linalg.norm(p2-p2_old)
        p1_old = p1
        p2_old = p2        
    return v - theta1 * div(p1,p2), i

if __name__ == "__main__":
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
    n_iterations = 1000
    theta1 = 1
    tol = 1
    
    u, i = _total_variation(image, theta1=theta1, maxit=n_iterations, tol=tol)
    
    
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(u, cmap="gray")
    ax[1].set_axis_off()
    title = "u - {} iterations".format(n_iterations)
    ax[1].set_title(title, fontsize=12)

    fig.tight_layout()
    plt.show()