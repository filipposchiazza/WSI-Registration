import openslide
import numpy as np


def slide_summary(slide, verbose=True):
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
    shape1 = np.shape(img1)
    shape2 = np.shape(img2)
    h, w = np.where(np.asarray(shape1[:2]) < np.asarray(shape2[:2]), shape1[:2], shape2[:2])
    print(f"The common shape is {h,w}")
    crop1 = img1[shape1[0]//2-h//2:shape1[0]//2+h//2, shape1[1]//2-w//2:shape1[1]//2+w//2]
    crop2 = img2[shape2[0]//2-h//2:shape2[0]//2+h//2, shape2[1]//2-w//2:shape2[1]//2+w//2]
    return crop1, crop2
    
