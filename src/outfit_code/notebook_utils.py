import numpy as np
import matplotlib.pyplot as plt

from utils.vis_utils import draw_kps


def draw_pcd_bg(bg_img, np_cam_kps):
    '''
    Draws the 2D point cloud `np_cam_kps` on top of the background image `bg_img`.
    '''
    
    non_black_pixels_mask = np.any(bg_img != [0, 0, 0], axis=-1)
    bg_img[non_black_pixels_mask] = [0, 0, 0]

    image = np.zeros((bg_img.shape[0], bg_img.shape[1], 3))

    draw_image = draw_kps(image, np_cam_kps, radius=1, thickness=2)
    # draw_image = cv2.resize(draw_image, bg_img.shape[:2])
    draw_image = draw_image.astype('uint')

    pred_pixels_mask = np.any(draw_image != [0, 0, 0], axis=-1)

    pred_img = np.ones_like(draw_image) * pred_pixels_mask[:,:,None] * 255
    pred_img = pred_img.astype('uint')
    pred_img = np.einsum('k,ijk->ijk', np.array([0, 255, 0]), pred_img)

    draw_image = 0.5 * bg_img + 0.5 * pred_img
    
    return draw_image


def show_nb(images, title=None, titles=None, n_cols=3):
    '''
    Conveniently shows the `images` in jupyter notebook.
    '''
    
    num_imgs = len(images)
    
    n_rows = num_imgs // n_cols
    
    fig = plt.figure(figsize=(n_cols*5, n_rows*5))
    
    if title is not None:
        plt.suptitle(title, fontsize=15)
    
    for i, img in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        if titles is not None:
            plt.title(titles[i], fontsize=15)
    
    plt.tight_layout()
    plt.show();
