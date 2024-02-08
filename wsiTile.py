import numpy as np
import openslide
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import tqdm
from skimage import transform
import os
import histomicstk as htk

#%%

class WSITile():
    
    def __init__(self, slide, level, top_left_coord, shape):
        """WSI tile
        
        Parameters
        ----------
        slide : Openslide.Slide
            Source slide to extract tiles.
        level : int
            Level in the pyramidal WSI structure (level 0 for higher resolution).
        top_left_coord : tuple
            Top left coordinates of the extracted slide, given as (y, x).
        shape : tuple
            Shape of the extracted tile, given as (h, w).

        Returns
        -------
        None.

        """
        self.slide = slide
        self.level = level
        self.top_left_coord = top_left_coord  # (y, x)
        self.shape = shape # (h, w)
        
        self.tile = self.extract_tile()
        self.tile_grayscale = np.round(np.mean(self.tile, axis=2)).astype(np.uint8)
        self.cyto = self.extract_cytoplasm()
        
    def extract_tile(self):
        factor = self.slide.level_downsamples[self.level]
        top_left_level_zero = (int(self.top_left_coord[1]*factor), int(self.top_left_coord[0]*factor))
        shape = (self.shape[1], self.shape[0])
        tile = self.slide.read_region(top_left_level_zero, self.level, shape)
        return np.asarray(tile)[:, :, :3]
        
    def extract_cytoplasm(self):
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(self.tile, 255)
        deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(self.tile, w_est, 255)
        channel = htk.preprocessing.color_deconvolution.find_stain_index(stain_color_map['eosin'], w_est)
        cyto = deconv_result.Stains[:, :, channel]
        return cyto
    



#%%

def find_matches(target, 
                 source, 
                 min_match_count=10, 
                 verbose=False, 
                 plot=False, 
                 target_to_show=None,
                 source_to_show=None,
                 plot_title=''):
    # Use SIFT for point detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(target, None)
    kp2, des2 = sift.detectAndCompute(source, None)
    
    # Match points according to Nearest Neighbours
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
            
    if len(good_matches) < min_match_count:
        print( "Not enough matches are found - {}/{}".format(len(good_matches), min_match_count) )
        return None
        
    target_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    source_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches ])
    
    # Use RANSAC to improve matches quality
    M, mask = cv2.findHomography(target_pts, source_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    good_matches_selected = []
    for i in range(len(good_matches)):
        if matchesMask[i] == 1:
            good_matches_selected.append(good_matches[i])
            
    
    if verbose == True:
        print(f"Number of initial matches: {len(matches)}")
        print(f"Number of matches after Lowe's test: {len(good_matches)}")
        print(f"Number of matches after RANSAC: {len(good_matches_selected)}")
        
    if plot == True:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        img = cv2.drawMatches(target_to_show, kp1, source_to_show, kp2, good_matches, None, **draw_params)
        plt.imshow(img, 'gray')
        plt.title(plot_title, color='red')
        plt.axis('off') 
        plt.show()
        
    # Extract points
    p = []
    q = []
    for i in range(len(good_matches_selected)):
        p.append(kp1[good_matches_selected[i].queryIdx].pt)
        q.append(kp2[good_matches_selected[i].trainIdx].pt)
        
    return np.array(p)[:, ::-1], np.array(q)[:, ::-1]  # from (x, y) to (y, x)


#%%

def evaluate_transformation_param(p, q, verbose=False):
    mean_p = np.mean(p, axis=0)
    mean_q = np.mean(q, axis=0)
    sigma_p = np.mean((p-mean_p)**2)
    sigma_q = np.mean((q-mean_q)**2)
    
    # Scale factor S
    S = np.sqrt(sigma_p/sigma_q)
    
    # Rotation matrix R
    C = np.matmul((q-mean_q).T, (p - mean_p))  # correlation matrix
    U, Sigma, Vt = np.linalg.svd(C)
    R = np.matmul(Vt.T, U.T)
        
    # Traslation vector T
    T = mean_p - np.matmul(R, S*mean_q)
    
    if verbose == True:
        print(f'Shape covariance matrix is {C.shape}')
        print(f'det(R) = {np.linalg.det(R)}')
        print(f'S = {S}')
        print(f'R = {R}')
        print(f'T = {T}')
    
    return S, R, T

#%%


def plot_aligned_points(p, q, S, R, T, point_size=30):
    mean_p = np.mean(p, axis=0)
    mean_q = np.mean(q, axis=0)
    q_transformed = np.matmul(R, (S*q).T).T + T
    mean_q_trasformed = np.mean(q_transformed, axis=0)
    
    # Plot before alignment
    plt.gca().invert_yaxis()
    plt.scatter(p[:,1], p[:,0], color='orange', s=point_size, label='p-points')
    plt.scatter(mean_p[1], mean_p[0], color='red', s=point_size, label='p-centroid')
    plt.scatter(q[:,1], q[:,0], color='green', marker='v', s=point_size, label='q-points')
    plt.scatter(mean_q[1], mean_q[0], color='blue', marker='v', s=point_size, label='q-centroid')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Before alignment', color='red')
    plt.show()
    
    # Plot after alignment
    plt.gca().invert_yaxis()
    plt.scatter(p[:,1], p[:,0], color='orange', s=point_size, label='p-points')
    plt.scatter(mean_p[1], mean_p[0], color='red', s=point_size, label='p-centroid')
    plt.scatter(q_transformed[:,1], q_transformed[:,0], color='green', marker='v', s=point_size, label='q-points')
    plt.scatter(mean_q_trasformed[1], mean_q_trasformed[0], color='blue', marker='v', s=point_size, label='q-centroid')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('After alignment', color='red')
    plt.show()
    
#%%

def image_registration(target, source, S, R, T):
    "(x', y') = R S (x, y) + T"
    h, w = source.shape

    aligned = np.zeros_like(source)

    for row in tqdm.tqdm(range(h)):
        for col in range(w):
            try:
                source_coord = np.array([row, col])
                row_new, col_new = np.matmul(R, S*source_coord.T) + T
                if row_new >= 0 and col_new >= 0:
                    aligned[round(row_new), round(col_new)] = source[row, col]
            except IndexError:
                continue
    
    return aligned


def merge_registered_imgs(target, source_reg):
    max_rows = max(target.shape[0], source_reg.shape[0])
    max_cols = max(target.shape[1], source_reg.shape[1])
    
    pad_rows_target = max_rows - target.shape[0]
    pad_cols_target = max_cols - target.shape[1]
    
    pad_rows_source = max_rows - source_reg.shape[0]
    pad_cols_source = max_cols - source_reg.shape[1]
    
    target_expanded = np.pad(target, ((0, pad_rows_target), (0, pad_cols_target)), mode='constant', constant_values=0)
    source_expanded = np.pad(source_reg, ((0, pad_rows_source), (0, pad_cols_source)), mode='constant', constant_values=0)
    
    r = target_expanded.astype(np.uint8)
    g = source_expanded.astype(np.uint8)
    b = np.zeros_like(r).astype(np.uint8)
    merged = cv2.merge((r, g, b))
    
    return merged
    
    
#%% Pyramidal alignment

def extract_target_tile(target_slide, level, top_left=(0,0), dims=(1024, 1024)):
    "Extract a tile from a given level"
    factor = target_slide.level_downsamples[level]
    top_left_level_zero = (int(top_left[0]*factor), int(top_left[1]*factor)) # (y, x)
    top_left_level_zero = (top_left_level_zero[1], top_left_level_zero[0]) # (x, y)
    dims = (dims[1], dims[0]) # (x, y)
    tile = target_slide.read_region(top_left_level_zero, level, dims)
    return np.asarray(tile)[:, :, :3]

    
def extract_target_points(top_left, dims):
    "Extract the four tile corners"
    top_right = (top_left[0], top_left[1] + dims[1])
    bottom_left = (top_left[0] + dims[0], top_left[1])
    bottom_right = (top_left[0] + dims[0], top_left[1] + dims[1])
    return top_left, top_right, bottom_left, bottom_right
    

def point_inverse_transformation(point, level, ref_level, S, R, T):
    "Transform the corner point in the source slide domain, through the parameters R, S, and T"
    delta = np.abs(level-ref_level)
    R_inv = np.linalg.inv(R)
    transformed_point = np.matmul(R_inv/S, (point - 2**delta * T).T)
    return np.round(transformed_point).astype(np.int64)

def extract_enlarged_source_tile(source_slide, source_points, level):
    top_left = source_points.min(axis=0)
    bottom_right = source_points.max(axis=0)
    dims = bottom_right - top_left
    source_tile = WSITile(slide=source_slide, 
                          level=level, 
                          top_left_coord=top_left, 
                          shape=dims)
    return source_tile
"""

def extract_enlarged_source_tile(source_slide, source_points, level):
    "Extract an enlarged area that contain the four source points"
    top_left = source_points.min(axis=0)#[::-1] # (x, y)
    bottom_right = source_points.max(axis=0)#[::-1] # (x, y)
    factor = source_slide.level_downsamples[level]
    top_left_level_zero = (int(top_left[0]*factor), int(top_left[1]*factor))
    dims = bottom_right - top_left
    source_tile = source_slide.read_region(top_left_level_zero, level, dims)
    return np.asarray(source_tile)[:, :, :3] 
"""

def rescale_points(points):
    points = np.asarray(points)
    return points - points.min(axis=0)

def plot_image_with_points(img, points, title, filename=None):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.scatter(points[:,1], points[:,0], c='red', s=10)
    if filename != None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
def homographic_transform(src_points, dst_points, src_image):
    "Perform homographic transformation between the two set of points"
    src_points = src_points[:, ::-1] # from (y, x) to (x, y)
    dst_points = dst_points[:, ::-1] # from (y, x) to (x, y)
    tform = transform.estimate_transform('projective', src_points, dst_points)
    tf_img = transform.warp(src_image, tform.inverse)
    tf_img = np.round(tf_img * 255).astype(np.uint8)
    return tf_img


def pyramid_alignment(trg_slide, src_slide, level, dims, top_left, R, S, T, ref_level=6, plot=True, save_folder=None):
    target_tile = extract_target_tile(trg_slide, level, top_left=top_left, dims=dims)
    target_points = np.asarray(extract_target_points(top_left, dims))
    source_points = point_inverse_transformation(target_points, level, ref_level, S, R, T).T
    source_tile = extract_enlarged_source_tile(src_slide, source_points, level)
    relative_target_points = rescale_points(target_points)
    relative_source_points = rescale_points(source_points)
    
    if save_folder == None:
        filename = None
    else:
        filename = os.path.join(save_folder, f'source_tile_level{level}.png')
    
    if plot == True:
        plot_image_with_points(source_tile.tile, relative_source_points, 'Source Tile', filename)
    
    if save_folder != None:
        filename = os.path.join(save_folder, f'target_tile_level{level}.png')
    plot_image_with_points(target_tile, relative_target_points, 'Target Tile', filename)
    
    tf_img = homographic_transform(relative_source_points, relative_target_points, source_tile.tile)
    if plot == True:
        plt.imshow(tf_img)
        plt.axis('off')
        plt.title('Source Homography Transformation')
        plt.show()
    if save_folder != None:
        filename = os.path.join(save_folder, f'homography_transf_level{level}.png')
        plt.savefig(filename, bbox_inches='tight')

    reg_image = plot_registered_imgs(target=np.round(np.mean(target_tile, axis=-1)).astype(np.uint8), 
                                     source_reg=np.round(np.mean(tf_img, axis=-1)).astype(np.uint8))
    return reg_image

#%%


def improve_alignment(target_slide, 
                      source_slide, 
                      R_ref, 
                      S_ref, 
                      T_ref, 
                      level, 
                      level_ref,
                      min_match_count,
                      tile_size=256,
                      var_threshold=300):
    
    # Divide the target slide at level l in tiles
    target_tile_set = divide_slide(target_slide, 
                                   level=level,
                                   tile_shape=(tile_size, tile_size))
    
    # Select tiles without background
    target_tile_set = remove_background_tiles(target_tile_set, var_threshold=var_threshold)
    
    # For each target tile, find the corresponding source tile
    source_tile_set = locate_source_tiles(source_slide, 
                                          target_tile_set, 
                                          level,
                                          level_ref,
                                          R_ref,
                                          S_ref,
                                          T_ref)
    
    # Use SIFT plus RANSAC to obtain p-points and q-points, directly on RGB image
    global_p_points, global_q_points = find_local_matches(target_tile_set,
                                                          source_tile_set, 
                                                          min_match_count=min_match_count,
                                                          verbose=True,
                                                          plot=True)
    S_new, R_new, T_new = evaluate_transformation_param(global_p_points, global_q_points)
    
    results_summary = {'target tile set' : target_tile_set,
                       'source tile set' : source_tile_set,
                       f'S level {level}': S_new,
                       f'R level {level}': R_new,
                       f'T level {level}': T_new,
                       'global target points': global_p_points,
                       'global source points': global_q_points}
    
    return results_summary
    

 


def divide_slide(slide, level, tile_shape):
    w, h = slide.level_dimensions[level]
    rows = h // tile_shape[0]
    cols = w // tile_shape[1]
    tile_set = []
    
    for x in range(cols+1):  # plus one added to keep into account bottom and right borders
        for y in range(rows+1):
            top_left_coord = (y * tile_shape[0], x * tile_shape[1])
            tile_set.append(WSITile(slide=slide, 
                                    level=level, 
                                    top_left_coord=top_left_coord, 
                                    shape=tile_shape))
            
    # Remove padding added to keep bottom and right tiles
    for i, tile in enumerate(tile_set):
        gray = tile.tile_grayscale
        non_zero_rows = np.any(gray != 0, axis=1)
        non_zero_cols = np.any(gray != 0, axis=0)
        tile_set[i].tile = tile.tile[non_zero_rows][:, non_zero_cols]
        tile_set[i].tile_grayscale = tile.tile_grayscale[non_zero_rows][:, non_zero_cols]
    return tile_set


def remove_background_tiles(tile_set, var_threshold):
    tiles_selected = []
    for tile in tile_set:
        var = np.var(tile.tile)
        if var > var_threshold:
            tiles_selected.append(tile)
    return tiles_selected


def locate_source_tiles(source_slide, target_tile_set, level, ref_level, R_ref, S_ref, T_ref):
    source_tile_set = []
    for target_tile in target_tile_set:
        # Extract 4 corner points from the target tile
        top_left, top_right, bottom_left, bottom_right = extract_target_points(top_left=target_tile.top_left_coord, 
                                                                               dims=target_tile.tile.shape)
        
        # Transform the target point in the source domain
        source_points = point_inverse_transformation(point=[top_left, top_right, bottom_left, bottom_right], 
                                                     level=level, 
                                                     ref_level=ref_level, 
                                                     S=S_ref, 
                                                     R=R_ref, 
                                                     T=T_ref)
        # Replace negative coordinates with zeros
        source_points = np.maximum(source_points, 0).T
        
        source_tile = extract_enlarged_source_tile(source_slide, 
                                                   source_points, 
                                                   level)
        
        
        source_tile_set.append(source_tile)
    
    return source_tile_set
        


def find_local_matches(target_tile_set, 
                       source_tile_set, 
                       min_match_count, 
                       verbose,
                       plot):
    global_p_points = []
    global_q_points = []

    for i in range(len(target_tile_set)):
        p_points, q_points = find_matches(target_tile_set[i].tile, 
                                          source_tile_set[i].tile,
                                          min_match_count=min_match_count,
                                          verbose=verbose,
                                          plot=plot,
                                          target_to_show=target_tile_set[i].tile,
                                          source_to_show=source_tile_set[i].tile,
                                          plot_title='SIFT detection + RANSAC')
        global_p_points.append(p_points + target_tile_set[i].top_left_coord)
        global_q_points.append(q_points + source_tile_set[i].top_left_coord)
    
    global_p_points = np.concatenate(global_p_points)
    global_q_points = np.concatenate(global_q_points)
    
    return global_p_points, global_q_points


    

    
        
        
    
    
            
        
 
        
