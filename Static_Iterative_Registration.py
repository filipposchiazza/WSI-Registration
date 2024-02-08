import numpy as np
import openslide
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import tqdm
from skimage import transform
import os
import histomicstk as htk

from wsiTile import WSITile
from wsiTile import find_matches, evaluate_transformation_param, plot_aligned_points 
from wsiTile import extract_target_points, point_inverse_transformation, extract_enlarged_source_tile
from wsiTile import plot_image_with_points, rescale_points, pyramid_alignment

from metrics import mutual_img_info

def global_registration(target_slide,
                        source_slide,
                        level):
    
    trg_shape = (target_slide.level_dimensions[level][1], target_slide.level_dimensions[level][0])
    trg_img = WSITile(slide=target_slide, 
                      level=level, 
                      top_left_coord=(0, 0), 
                      shape=trg_shape)
    
    src_shape = (source_slide.level_dimensions[level][1], source_slide.level_dimensions[level][0])
    src_img = WSITile(slide=source_slide, 
                      level=level, 
                      top_left_coord=(0, 0), 
                      shape=src_shape)
    
    
    trg_points, src_points = find_matches(target=trg_img.cyto, 
                                          source=src_img.cyto)
    
    
    S, R, T = evaluate_transformation_param(p=trg_points, 
                                            q=src_points)
    
    
    return trg_points, src_points, S, R, T




def locate_source_tiles(source_slide, target_tile, level, ref_level, R_ref, S_ref, T_ref):
    
    top_left, top_right, bottom_left, bottom_right = extract_target_points(top_left=target_tile.top_left_coord, 
                                                                           dims=target_tile.tile.shape)
    
    src_points = point_inverse_transformation(point=[top_left, top_right, bottom_left, bottom_right], 
                                              level=level, 
                                              ref_level=ref_level, 
                                              S=S_ref, 
                                              R=R_ref, 
                                              T=T_ref)
    # Replace negative coordinates with zeros
    src_points = np.maximum(src_points, 0).T
    
    src_tile = extract_enlarged_source_tile(source_slide, 
                                            src_points, 
                                            level)
    return src_tile, src_points
    


















#%%

TARGET_SLIDE = openslide.open_slide("/home/filippo/Scrivania/1484/20-1081/HE_2.ndpi")
SOURCE_SLIDE = openslide.open_slide("/home/filippo/Scrivania/1484/20-1081/Giemsa.ndpi")
START_LEVEL = 6
START_TOP_LEFT = (620, 860) # Level 5, (y, x)
TILE_SIZE = (512, 512)

trg_points, src_points, S, R, T = global_registration(target_slide=TARGET_SLIDE, 
                                                      source_slide=SOURCE_SLIDE, 
                                                      level=START_LEVEL)

plot_aligned_points(trg_points, src_points, S, R, T, point_size=15)

trg_tile5 = WSITile(slide=TARGET_SLIDE, 
                    level=5, 
                    top_left_coord=START_TOP_LEFT, 
                    shape=TILE_SIZE)

src_tile5, src_corners5 = locate_source_tiles(source_slide=SOURCE_SLIDE, 
                                             target_tile=trg_tile5, 
                                             level=5, 
                                             ref_level=START_LEVEL, 
                                             R_ref=R, 
                                             S_ref=S, 
                                             T_ref=T)

merged_before = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           5, 
                           TILE_SIZE, 
                           START_TOP_LEFT, 
                           R, 
                           S, 
                           T)

#%% Register level 5

trg_points5, src_points5 = find_matches(trg_tile5.tile,
                                        src_tile5.tile,
                                        min_match_count=8,
                                        plot=True,
                                        target_to_show=trg_tile5.tile,
                                        source_to_show=src_tile5.tile)

global_trg_points5 = trg_points5 + trg_tile5.top_left_coord
global_src_points5 = src_points5 + src_tile5.top_left_coord

S5, R5, T5 = evaluate_transformation_param(p=global_trg_points5, 
                                           q=global_src_points5)

plot_aligned_points(global_trg_points5, global_src_points5, S5, R5, T5)

merged_after = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           5, 
                           TILE_SIZE, 
                           START_TOP_LEFT, 
                           R5, 
                           S5, 
                           T5/2)


r_before = merged_before[:, :, 0]
g_before = merged_before[:, :, 1]
r_after = merged_after[:, :, 0]
g_after = merged_after[:, :, 1]
mutual_img_info(r_before, g_before, plot_log_hist=False)
mutual_img_info(r_after, g_after, plot_log_hist=False)

#%% Register level 4

merged_before = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           4, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 2), 
                           R, 
                           S, 
                           T)

trg_tile4 = WSITile(slide=TARGET_SLIDE, 
                    level=4, 
                    top_left_coord=np.dot(START_TOP_LEFT, 2), 
                    shape=TILE_SIZE)

src_tile4, src_corners4 = locate_source_tiles(source_slide=SOURCE_SLIDE, 
                                              target_tile=trg_tile4, 
                                              level=4, 
                                              ref_level=5, 
                                              R_ref=R5, 
                                              S_ref=S5, 
                                              T_ref=T5)

trg_points4, src_points4 = find_matches(trg_tile4.tile,
                                        src_tile4.tile,
                                        min_match_count=8,
                                        plot=True,
                                        target_to_show=trg_tile4.tile,
                                        source_to_show=src_tile4.tile)

global_trg_points4 = trg_points4 + trg_tile4.top_left_coord
global_src_points4 = src_points4 + src_tile4.top_left_coord

S4, R4, T4 = evaluate_transformation_param(p=global_trg_points4, 
                                           q=global_src_points4)

plot_aligned_points(global_trg_points4, global_src_points4, S4, R4, T4)

merged_after = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           4, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 2), 
                           R4, 
                           S4, 
                           T4/4)


#%% Register level 3

merged_before = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           3, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 4), 
                           R, 
                           S, 
                           T)



trg_tile3 = WSITile(slide=TARGET_SLIDE, 
                    level=3, 
                    top_left_coord=np.dot(START_TOP_LEFT, 4), 
                    shape=TILE_SIZE)

src_tile3, src_corners3 = locate_source_tiles(source_slide=SOURCE_SLIDE, 
                                              target_tile=trg_tile3, 
                                              level=3, 
                                              ref_level=4, 
                                              R_ref=R4, 
                                              S_ref=S4, 
                                              T_ref=T4)

trg_points3, src_points3 = find_matches(trg_tile3.tile,
                                        src_tile3.tile,
                                        min_match_count=8,
                                        plot=True,
                                        target_to_show=trg_tile3.tile,
                                        source_to_show=src_tile3.tile)

global_trg_points3 = trg_points3 + trg_tile3.top_left_coord
global_src_points3 = src_points3 + src_tile3.top_left_coord

S3, R3, T3 = evaluate_transformation_param(p=global_trg_points3, 
                                           q=global_src_points3)

plot_aligned_points(global_trg_points3, global_src_points3, S3, R3, T3)

merged_after = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           3, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 4), 
                           R3, 
                           S3, 
                           T3/8)

#%% Register level 2

merged_before = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           2, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 8), 
                           R, 
                           S, 
                           T)




trg_tile2 = WSITile(slide=TARGET_SLIDE, 
                    level=2, 
                    top_left_coord=np.dot(START_TOP_LEFT, 8), 
                    shape=TILE_SIZE)

src_tile2, src_corners2 = locate_source_tiles(source_slide=SOURCE_SLIDE, 
                                              target_tile=trg_tile2, 
                                              level=2, 
                                              ref_level=3, 
                                              R_ref=R3, 
                                              S_ref=S3, 
                                              T_ref=T3)

trg_points2, src_points2 = find_matches(trg_tile2.tile,
                                        src_tile2.tile,
                                        min_match_count=8,
                                        plot=True,
                                        target_to_show=trg_tile2.tile,
                                        source_to_show=src_tile2.tile)

global_trg_points2 = trg_points2 + trg_tile2.top_left_coord
global_src_points2 = src_points2 + src_tile2.top_left_coord

S2, R2, T2 = evaluate_transformation_param(p=global_trg_points2, 
                                           q=global_src_points2)

plot_aligned_points(global_trg_points2, global_src_points2, S2, R2, T2)

merged_after = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           2, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 8), 
                           R2, 
                           S2, 
                           T2/16)


#%% Register level 1

merged_before= pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           1, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 16), 
                           R, 
                           S, 
                           T)

merged_after = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           1, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 16), 
                           R2, 
                           S2, 
                           T2/16)


trg_tile1 = WSITile(slide=TARGET_SLIDE, 
                    level=1, 
                    top_left_coord=np.dot(START_TOP_LEFT, 16), 
                    shape=TILE_SIZE)

src_tile1, src_corners1 = locate_source_tiles(source_slide=SOURCE_SLIDE, 
                                              target_tile=trg_tile1, 
                                              level=1, 
                                              ref_level=2, 
                                              R_ref=R2, 
                                              S_ref=S2, 
                                              T_ref=T2)

trg_tile1.extract_cytoplasm()
src_tile1.extract_cytoplasm()

trg_points1, src_points1 = find_matches(trg_tile1.cyto,
                                        src_tile1.cyto,
                                        min_match_count=8,
                                        plot=True,
                                        target_to_show=trg_tile1.tile,
                                        source_to_show=src_tile1.tile)

global_trg_points1 = trg_points1 + trg_tile1.top_left_coord
global_src_points1 = src_points1 + src_tile1.top_left_coord

S1, R1, T1 = evaluate_transformation_param(p=global_trg_points1, 
                                           q=global_src_points1)

plot_aligned_points(global_trg_points1, global_src_points1, S1, R1, T1)

merged = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           2, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 8), 
                           R2, 
                           S2, 
                           T2/16)

#%% Register level 0

merged_before = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           0, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 32), 
                           R, 
                           S, 
                           T)


merged_after = pyramid_alignment(TARGET_SLIDE, 
                           SOURCE_SLIDE, 
                           0, 
                           TILE_SIZE, 
                           np.dot(START_TOP_LEFT, 32), 
                           R2, 
                           S2, 
                           T2/16)








import matplotlib.pyplot as plt
import os
from PIL import Image

# Definisci il percorso della cartella delle immagini
folder_path = "/home/filippo/Documenti/Ricerca/Presentazioni/Dicembre23Registration(2)/Images/"

# Elenco delle immagini nella cartella
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])

# Numero di righe e colonne nel collage
rows = 3
cols = 2

# Crea una figura con sottoplot
fig, axs = plt.subplots(rows, cols, figsize=(8, 8))

# Ciclo attraverso le immagini e posizionale sui sottoplot
for i in range(rows):
    for j in range(cols):
        axs[i, j].axis('off')
        if i == 1 and j ==3:
            break
        img_path = os.path.join(folder_path, image_files[i * cols + j])
        img = Image.open(img_path)
        axs[i, j].imshow(img)

# Salva il collage in un file
#plt.savefig("output_collage.png", bbox_inches='tight')

# Mostra il collage
plt.show()







