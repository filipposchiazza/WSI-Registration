from modules.wsiTile import WSITile
from modules.metrics import mutual_img_info
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from skimage import transform
import os



class WSIRegister():
    
    def __init__(self, target_slide, source_slide):
        """Class to register two Whole Slide Images
        
        Parameters
        ----------
        target_slide : openslide.OpenSlide
            Target slide to register
        source_slide : openslide.OpenSlide
            Source slide to register
        """
        self.target_slide = target_slide
        self.source_slide = source_slide
        
     
    @staticmethod
    def find_matches_with_SIFT_and_RANSAC(target, 
                                          source, 
                                          min_match_count=10, 
                                          verbose=False, 
                                          plot=False, 
                                          target_to_show=None,
                                          source_to_show=None,
                                          plot_title=''):
        """Find matches between two images using SIFT and RANSAC
        
        Parameters
        ----------
        target : numpy.ndarray
            Target image
        source : numpy.ndarray
            Source image
        min_match_count : int, optional
            Minimum number of matches to consider, by default 10
        verbose : bool, optional
            If True, print the number of matches, by default False
        plot : bool, optional
            If True, plot the matches, by default False
        target_to_show : numpy.ndarray, optional
            Target image to show in the plot, by default None
        source_to_show : numpy.ndarray, optional
            Source image to show in the plot, by default None
        plot_title : str, optional
            Title of the plot
        
        Returns
        -------
        p : numpy.ndarray
            Points in the target image expressed as (y, x)
        q : numpy.ndarray
            Points in the source image expressed as (y, x)
        """
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
            error_message = "Not enough matches are found - {}/{}".format(len(good_matches), min_match_count)
            raise ValueError(error_message)
            
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
    
    
    
    @staticmethod
    def evaluate_transformation_param(p, q, verbose=False):
        """Evaluate the transformation parameters S, R, and T
        
        Parameters
        ----------
        p : numpy.ndarray
            Points in the target image expressed as (y, x)
        q : numpy.ndarray
            Points in the source image expressed as (y, x)
        verbose : bool, optional
            If True, print the transformation parameters, by default False
        
        Returns
        -------
        S : float
            Scale factor
        R : numpy.ndarray
            Rotation matrix
        T : numpy.ndarray
            Translation vector
        """
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
    
    

    @staticmethod
    def plot_aligned_points(p, q, S, R, T, point_size=30):
        """Plot the points before and after alignment
        
        Parameters
        ----------
        p : numpy.ndarray
            Points in the target image expressed as (y, x)
        q : numpy.ndarray
            Points in the source image expressed as (y, x)
        S : float
            Scale factor
        R : numpy.ndarray
            Rotation matrix
        T : numpy.ndarray
            Translation vector
        point_size : int, optional
            Size of the points, by default 30

        Returns
        -------
        None
        """
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
        
        
        
    @staticmethod
    def image_registration(target, source, S, R, T):
        """Perform the pointwise trasformation (x', y') = R S (x, y) + T
        
        Parameters
        ----------
        target : numpy.ndarray
            Target image
        source : numpy.ndarray
            Source image
        S : float
            Scale factor
        R : numpy.ndarray
            Rotation matrix
        T : numpy.ndarray   
            Translation vector
        
        Returns
        -------
        aligned : numpy.ndarray
            Aligned source image
        """
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
    
    
    
    @staticmethod
    def merge_registered_imgs(target, source_reg):
        """Merge the registered images

        Parameters
        ----------
        target : numpy.ndarray
            Target image   
        source_reg : numpy.ndarray
            Registered source image
        
        Returns
        -------
        merged : numpy.ndarray
            Merged image, where the target is the red channel and the source is the green channel
        """
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
    
    

    # Pyramidal alignment
    def extract_target_tile(self, level, top_left=(0,0), dims=(1024, 1024)):
        """Extract a tile from the target slide
        
        Parameters
        ----------
        level : int
            Level of the slide
        top_left : tuple, optional
            Top left coordinates of the tile, by default (0,0) and expressed as (y, x)
        dims : tuple, optional
            Dimensions of the tile, by default (1024, 1024) and expressed as (y, x)
        
        Returns
        -------
        tile : WSITile
            Tile object extracted from the target slide
        """
        tile = WSITile(slide=self.target_slide, 
                       level=level, 
                       top_left_coord=top_left, 
                       shape=dims)
        return tile
    
    

    @staticmethod
    def extract_target_points(top_left, dims):
        """Extract the four corner points of the tile
        
        Parameters
        ----------
        top_left : tuple
            Top left coordinates of the tile, expressed as (y, x)
        dims : tuple
            Dimensions of the tile, expressed as (y, x)
            
        Returns
        -------
        points : numpy.ndarray
            Array of the four corner points
        """
        top_right = (top_left[0], top_left[1] + dims[1])
        bottom_left = (top_left[0] + dims[0], top_left[1])
        bottom_right = (top_left[0] + dims[0], top_left[1] + dims[1])
        return np.asarray([top_left, top_right, bottom_left, bottom_right])
    
    

    @staticmethod
    def point_inverse_transformation(point, level, ref_level, S, R, T):
        """Transform the corner point in the source slide domain, through the parameters R, S, and T
        
        Parameters
        ----------
        point : numpy.ndarray
            Point to transform
        level : int
            Level of the slide
        ref_level : int
            Reference level where the global trasformation parameters S, R, and T have been evaluated
        S : float
            Scale factor
        R : numpy.ndarray
            Rotation matrix
        T : numpy.ndarray
            Translation vector
        
        Returns
        -------
        transformed_point : numpy.ndarray
            Transformed point
        """
        delta = np.abs(level-ref_level)
        R_inv = np.linalg.inv(R)
        transformed_point = np.matmul(R_inv/S, (point - 2**delta * T).T)
        return np.round(transformed_point).astype(np.int64)
    
    

    def extract_enlarged_source_tile(self, source_points, level):
        """Extract an enlarged tile from the source slide that contains the transformed points
        
        Parameters
        ----------
        source_points : numpy.ndarray
            Points in the source slide domain
        level : int
            Level of the slide
        
        Returns
        -------
        source_tile : WSITile
            Enlarged tile object extracted from the source slide
        """
        top_left = source_points.min(axis=0)
        bottom_right = source_points.max(axis=0)
        dims = bottom_right - top_left
        source_tile = WSITile(slide=self.source_slide, 
                              level=level, 
                              top_left_coord=top_left, 
                              shape=dims)
        return source_tile
    
    

    @staticmethod
    def rescale_points(points):
        """Rescale points to the origin of the tile
        
        Parameters
        ----------
        points : numpy.ndarray
            Points to rescale
        
        Returns
        -------
        numpy.ndarray
            Rescaled points
        """
        points = np.asarray(points)
        return points - points.min(axis=0)
    

    
    @staticmethod
    def plot_image_with_points(img, points, title, filename=None, axis='off', point_color='red', point_size=10):
        """Plot the image with points
        
        Parameters
        ----------
        img : numpy.ndarray
            Image to plot
        points : numpy.ndarray
            Points to plot
        title : str
            Title of the plot
        filename : str, optional
            Name of the file where to save the plot, by default None
        axis : str, optional
            Axis of the plot, by default 'off'
        point_color : str, optional
            Color of the points, by default 'red'
        point_size : int, optional
            Size of the points, by default 10
            
        Returns
        -------
        None
        """
        plt.imshow(img)
        plt.axis(axis)
        plt.title(title)
        plt.scatter(points[:,1], points[:,0], c=point_color, s=point_size)
        if filename != None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        
        

    @staticmethod
    def homographic_transform(src_points, dst_points, src_image):
        """Perform homographic transformation between the two set of points
        
        Parameters
        ----------
        src_points : numpy.ndarray
            Points in the source image
        dst_points : numpy.ndarray
            Points in the destination image
        src_image : numpy.ndarray
            Source image to transform
        
        Returns
        -------
        tf_img : numpy.ndarray
            Transformed image
        """
        src_points = src_points[:, ::-1] # from (y, x) to (x, y)
        dst_points = dst_points[:, ::-1] # from (y, x) to (x, y)
        tform = transform.estimate_transform('projective', src_points, dst_points)
        tf_img = transform.warp(src_image, tform.inverse)
        tf_img = np.round(tf_img * 255).astype(np.uint8)
        return tf_img
    
    

    def pyramid_alignment(self, level, dims, top_left, R, S, T, ref_level=6, plot=True, save_folder=None):
        """Perform the alignment of the source image to the target image through a pyramidal approach

        Parameters
        ----------
        level : int
            Level of the slide
        dims : tuple
            Dimensions of the tile expressed as (y, x)
        top_left : tuple
            Top left coordinates of the tile, expressed as (y, x)
        R : numpy.ndarray
            Rotation matrix
        S : float
            Scale factor
        T : numpy.ndarray
            Translation vector
        ref_level : int, optional
            Reference level where the global trasformation parameters S, R, and T have been evaluated, by default 6
        plot : bool, optional
            If True, plot the images, by default True
        save_folder : str, optional
            Folder where to save the images, by default None
        
        Returns
        -------
        reg_img : numpy.ndarray
            Registered image
        """
        
        target_tile = self.extract_target_tile(level, 
                                               top_left=top_left, 
                                               dims=dims)
        target_points = self.extract_target_points(top_left, dims)
        source_points = self.point_inverse_transformation(target_points, level, ref_level, S, R, T).T
        source_tile = self.extract_enlarged_source_tile(source_points, level)
        relative_target_points = self.rescale_points(target_points)
        relative_source_points = self.rescale_points(source_points)
        tf_img = self.homographic_transform(relative_source_points, relative_target_points, source_tile.tile)
        reg_img = self.merge_registered_imgs(target=np.round(np.mean(target_tile.tile, axis=-1)).astype(np.uint8), 
                                             source_reg=np.round(np.mean(tf_img, axis=-1)).astype(np.uint8))
        
        if save_folder == None:
            filename1 = None
            filename2 = None
        else:
            filename1 = os.path.join(save_folder, f'target_tile_level{level}.png')
            filename2 = os.path.join(save_folder, f'source_tile_level{level}.png')
            filename3 = os.path.join(save_folder, f'homography_transf_level{level}.png')
            
        if plot == True:
            self.plot_image_with_points(target_tile.tile, relative_target_points, filename=filename1, title='Target Tile')
            self.plot_image_with_points(source_tile.tile, relative_source_points, filename=filename2, title='Source Tile')
            
            plt.imshow(tf_img)
            plt.axis('off')
            plt.title('Source Homography Transformation')
            if save_folder != None:
                plt.savefig(filename3, bbox_inches='tight')
            plt.show()
            
            plt.imshow(reg_img)
            plt.axis('off')
            plt.title('Registered images')

        return reg_img
    
    

    # Iterative method
    def global_registration(self, level, min_match_count):
        """Perform the global registration of the two slides
        
        Parameters
        ----------
        level : int
            Level of the slide where to perform the global registration
        min_match_count : int
            Minimum number of point matches to consider
        
        Returns
        -------
        trg_points : numpy.ndarray
            Points in the target image expressed as (y, x)
        src_points : numpy.ndarray
            Points in the source image expressed as (y, x)
        S : float
            Scale factor
        R : numpy.ndarray
            Rotation matrix
        T : numpy.ndarray
            Translation vector
        """
        
        trg_shape = (self.target_slide.level_dimensions[level][1], self.target_slide.level_dimensions[level][0])
        trg_img = WSITile(slide=self.target_slide, 
                          level=level, 
                          top_left_coord=(0, 0), 
                          shape=trg_shape)
        
        src_shape = (self.source_slide.level_dimensions[level][1], self.source_slide.level_dimensions[level][0])
        src_img = WSITile(slide=self.source_slide, 
                          level=level, 
                          top_left_coord=(0, 0), 
                          shape=src_shape)
        
        try:
            trg_points, src_points = self.find_matches_with_SIFT_and_RANSAC(target=trg_img.cyto, 
                                                                            source=src_img.cyto,
                                                                            min_match_count=min_match_count)
        except ValueError as e:
            print(f'Error: {e}')
            print('If possible, try to reduce min_match_count')
            return -1
        
        
        S, R, T = self.evaluate_transformation_param(p=trg_points, 
                                                     q=src_points)
        
        
        return trg_points, src_points, S, R, T
    
    
    
    def locate_source_tiles(self, target_tile, level, ref_level, R, S, T):
        """Locate the source tile in the source slide domain
        
        Parameters
        ----------
        target_tile : WSITile
            Target tile
        level : int
            Level of the slide where to perform the  local registration
        ref_level : int
            Reference level where the global trasformation parameters S, R, and T have been evaluated
        R : numpy.ndarray
            Rotation matrix
        S : float
            Scale factor
        T : numpy.ndarray
            Translation vector
            
        Returns
        -------
        src_tile : WSITile
            Source tile
        src_points : numpy.ndarray
            Points in the source image expressed as (y, x)
        """
        
        top_left, top_right, bottom_left, bottom_right = self.extract_target_points(top_left=target_tile.top_left_coord, 
                                                                                    dims=target_tile.tile.shape)
        
        src_points = self.point_inverse_transformation(point=[top_left, top_right, bottom_left, bottom_right], 
                                                       level=level, 
                                                       ref_level=ref_level, 
                                                       S=S, 
                                                       R=R, 
                                                       T=T)
        # Replace negative coordinates with zeros
        src_points = np.maximum(src_points, 0).T
        
        src_tile = self.extract_enlarged_source_tile(src_points, level)

        return src_tile, src_points
    
    
    
    def local_iterative_registration(self, 
                                     level_start, 
                                     level_stop, 
                                     top_left_start, 
                                     tile_size, 
                                     min_match_count=8, 
                                     plot=False):
        """Perform the local iterative registration of the two slides
        
        Parameters
        ----------
        level_start : int
            Level of the slide where to start the local registration
        level_stop : int
            Level of the slide where to stop the local registration
        top_left_start : tuple
            Top left coordinates of the tile, expressed as (y, x)
        tile_size : tuple
            Dimensions of the tile, expressed as (y, x)
        min_match_count : int, optional
            Minimum number of point matches to consider, by default 8
        plot : bool, optional
            If True, plot the resulting images, by default False
            
        Returns
        -------
        parameters : dict
            Dictionary of the trasformation parameters evaluated at each level
        """
        # first global registration
        global_trg_points, global_src_points, S, R, T = self.global_registration(level=level_start,
                                                                                 min_match_count=min_match_count)
        if plot == True:
            self.plot_aligned_points(global_trg_points, global_src_points, S, R, T, point_size=15)
            
        parameters = {f'level_{level_start}' : [S, R, T]}
        factor = 2
        reference_level = level_start
        for level in reversed(range(level_stop, level_start)):
            # rescale the tile top left coordinates to the current  resolution
            top_left_coord = np.dot(top_left_start, factor)
            trg_tile = WSITile(slide=self.target_slide, 
                               level=level, 
                               top_left_coord=top_left_coord, 
                               shape=tile_size)
        
            src_tile, src_corners = self.locate_source_tiles(target_tile=trg_tile, 
                                                             level=level, 
                                                             ref_level=reference_level, 
                                                             R=R, 
                                                             S=S, 
                                                             T=T)
            try:
                trg_points, src_points = self.find_matches_with_SIFT_and_RANSAC(trg_tile.tile,
                                                                                src_tile.tile,
                                                                                min_match_count=min_match_count,
                                                                                plot=plot,
                                                                                target_to_show=trg_tile.tile,
                                                                                source_to_show=src_tile.tile)
            except ValueError as e:
                print(f'Error: {e}\n','If possible, try to reduce min_match_count')
                break
        
            rescaled_trg_points = trg_points + trg_tile.top_left_coord
            rescaled_src_points = src_points + src_tile.top_left_coord
        
            S_up, R_up, T_up = self.evaluate_transformation_param(p=rescaled_trg_points, 
                                                                  q=rescaled_src_points)
            
            # Verify that the new transformations are not to far from the older ones
            th_S = (np.abs(S-S_up) / S) * 100
            th_R = (np.linalg.norm(R-R_up) / np.linalg.norm(R)) * 100
            th_T = (np.linalg.norm(T-np.asarray(T_up)/2) / np.linalg.norm(T)) * 100
            
            if th_S > 12 or th_R > 8 or th_T > 30:
                break
            
            S = S_up
            R = R_up
            T = T_up
            parameters[f'level_{level}'] = [S, R, T]
        
            if plot == True:
                self.plot_aligned_points(rescaled_trg_points, rescaled_src_points, S, R, T, point_size=15)
                
        
            factor *= 2
            reference_level = level
        
        return parameters
    
    
    
    def apply_iterative_registration(self, level, ref_level, top_left_start, tile_size, parameters, plot=False):
        """Apply the iterative registration to the source slide
        
        Parameters
        ----------
        level : int
            Level of the slide where to apply the iterative registration
        ref_level : int
            Reference level where the global trasformation parameters S, R, and T have been evaluated
        top_left_start : tuple
            Top left coordinates of the tile, expressed as (y, x)
        tile_size : tuple
            Dimensions of the tile, expressed as (y, x)
        parameters : dict
            Dictionary of the trasformation parameters evaluated at each level
        plot : bool, optional
            If True, plot the resulting images, by default False
        
        Returns
        -------
        reg_img : numpy.ndarray
            Registered image
        """
        
        factor = 2**(ref_level - level)
        top_left = np.dot(top_left_start, factor)
        
        try:
            S, R, T = parameters[f'level_{level}']
            T_rescaled = T/factor
        except KeyError:
            S, R, T = parameters[list(parameters.keys())[-1]]
            T_factor = 2 ** (len(parameters) - 1)
            T_rescaled = T/T_factor
        
        reg_img = self.pyramid_alignment(level=level, 
                                         ref_level=ref_level,
                                         dims=tile_size, 
                                         top_left=top_left, 
                                         R=R, 
                                         S=S, 
                                         T=T_rescaled,
                                         plot=plot)
        
        return reg_img



    def stack_static_registration(self, ref_level, top_left_start, tile_size, S, R, T):
        """Stack the images obtained through the static registration
        
        Parameters
        ----------
        ref_level : int
            Reference level where the global trasformation parameters S, R, and T have been evaluated
        top_left_start : tuple
            Top left coordinates of the tile, expressed as (y, x)
        tile_size : tuple
            Dimensions of the tile, expressed as (y, x)
        S : float
            Scale factor
        R : numpy.ndarray
            Rotation matrix
        T : numpy.ndarray
            Translation vector
        
        Returns
        -------
        reg_imgs : list
            List of the registered images   
        """
        reg_imgs = []
        factor = 2
        top_left = top_left_start
        for level in reversed(range(ref_level)):
            top_left = np.dot(top_left, factor)
    
            reg_img = self.pyramid_alignment(level=level, 
                                             dims=tile_size, 
                                             top_left=top_left, 
                                             R=R, 
                                             S=S, 
                                             T=T,
                                             plot=False)
            reg_imgs.append(reg_img)
        return reg_imgs
    

    
    def stack_iterative_registration(self, ref_level, top_left_start, tile_size, parameters):
        """Stack the images obtained through the iterative registration
        
        Parameters
        ----------
        ref_level : int
            Reference level where the global trasformation parameters S, R, and T have been evaluated
        top_left_start : tuple
            Top left coordinates of the tile, expressed as (y, x)
        tile_size : tuple
            Dimensions of the tile, expressed as (y, x)
        parameters : dict
            Dictionary of the trasformation parameters evaluated at each level
        
        Returns
        -------
        reg_imgs : list
            List of the registered images"""

        reg_imgs = []
        
        for level in reversed(range(ref_level)):
            reg_img = self.apply_iterative_registration(level=level, 
                                                        ref_level=ref_level, 
                                                        top_left_start=top_left_start, 
                                                        tile_size=tile_size, 
                                                        parameters=parameters)
            
            reg_imgs.append(reg_img)
            
        return reg_imgs
    
    
    
    @staticmethod
    def mii_evaluation(img):
        """Evaluate the Mutual Image Information between the red and green channels of an image
        
        Parameters
        ----------
        img : numpy.ndarray
            Image to evaluate
        
        Returns
        -------
        mii : float
            Mutual Image Information
        """
        r = img[:, :, 0]
        g = img[:, :, 1]
        mii = mutual_img_info(r, g, plot_log_hist=False)
        return mii
        
        
        

        
                
        
        
    











