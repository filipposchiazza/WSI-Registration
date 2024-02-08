import openslide
import numpy as np


def slide_summary(slide, verbose=True):
    """Prints a summary of the slide properties
    
    Parameters
    ----------
    slide : openslide.OpenSlide
        Slide to summarize
    verbose : bool, optional
        If True, print the summary, by default True
        
    Returns
    -------
    slide_prop : dict
        Properties of the slide
    objective : float
        Objective power of the slide    
    dimensions : list
        Dimensions of the slide at different levels
    factors : list
        Downsample factors at different levels
    """
    slide_prop = slide.properties
    objective = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    dimensions = slide.level_dimensions
    factors = slide.level_downsamples
    
    if verbose:
        print(f"the objective power is {objective}")
        print(f"Dimensions are: {dimensions}")
        print(f"Factors are: {factors}")
        
    return slide_prop, objective, dimensions, factors    



def crop_center(img1, img2):
    """Crop the center of two images to the same size

    Parameters
    ----------
    img1 : numpy.ndarray
        First image to crop
    img2 : numpy.ndarray
        Second image to crop
    
    Returns
    -------
    crop1 : numpy.ndarray
        Cropped first image
    crop2 : numpy.ndarray
        Cropped second image
    """
    shape1 = np.shape(img1)
    shape2 = np.shape(img2)
    h, w = np.where(np.asarray(shape1[:2]) < np.asarray(shape2[:2]), shape1[:2], shape2[:2])
    print(f"The common shape is {h,w}")
    crop1 = img1[shape1[0]//2-h//2:shape1[0]//2+h//2, shape1[1]//2-w//2:shape1[1]//2+w//2]
    crop2 = img2[shape2[0]//2-h//2:shape2[0]//2+h//2, shape2[1]//2-w//2:shape2[1]//2+w//2]
    return crop1, crop2
    
