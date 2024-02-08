import numpy as np
import matplotlib.pyplot as plt



def mutual_img_info(img1, img2, plot_log_hist=True):
    """Calculate Mutual Information between img1 and img2

    Parameters
    ----------
    img1 : numpy.ndarray
        First image to compare
    img2 : numpy.ndarray
        Second image to compare
    plot_log_hist : bool, optional
        If True, plot the log of the joint histogram, by default True

    Returns
    -------
    float
        Mutual Information between img1 and img2
    """
    hist_2d, _, _ = np.histogram2d(img1.ravel(), 
                                   img2.ravel(),
                                   bins=255)
    if plot_log_hist == True:
        hist_2d_log = np.zeros(hist_2d.shape)
        non_zeros = (hist_2d != 0)
        hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        plt.imshow(hist_2d_log.T, origin='lower')
        
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))



def norm_cross_corr(img1, img2):
    """Calculate thr normalized cross-correlation between img1 and img2

    Parameters
    ----------
    img1 : numpy.ndarray
        First image to compare
    img2 : numpy.ndarray
        Second image to compare
    
    Returns
    -------
    float
        Normalized Cross-Correlation between img1 and img2
    """
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    diff1 = img1 - mean1
    diff2 = img2 - mean2
    diff1_squared = np.power(diff1, 2)
    diff2_squared = np.power(diff2, 2)
    num = np.sum(diff1 * diff2)
    den = np.sqrt(np.sum(diff1_squared) * np.sum(diff2_squared))
    return num/den



def mse(img1, img2):
    """Calculate the Mean Squared Error between img1 and img2

    Parameters
    ----------
    img1 : numpy.ndarray
        First image to compare
    img2 : numpy.ndarray
        Second image to compare
    
    Returns
    -------
    float
        Mean Squared Error between img1 and img2
    """
    h, w = np.shape(img1)
    n = h * w
    return np.sum(np.power(img1 - img2, 2)) / n

