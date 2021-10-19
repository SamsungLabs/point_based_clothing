import numpy as np
import cv2


def draw_kps(img, keypoints, radius=5, thickness=10, color=(0, 255, 0)):
    '''
    Draws keypoints and returns the image with them visualized.
    '''
    
    kp_img = img.copy()
    for kp_id in range(len(keypoints)):
        x, y = keypoints[kp_id]
        x, y = min(max(0, int(x)), kp_img.shape[1]), min(max(0, int(y)), kp_img.shape[0])
        kp_img = cv2.circle(kp_img, (x, y), radius, color, thickness)
    return kp_img


def visualize_code_fitting(bkg_image, pcd, video_writer, pids=None):
    '''
    Draws the point cloud on top of the background image and saves this frame to a video file.
    
    Args:
        bkg_image (`torch.FloatTensor`): background image on top of which to draw the point cloud.
        pcd (`torch.FloatTensor`): point cloud of size `(batch_size, pcd_size, 2)`.
        video_writer (`cv2.VideoWriter`): video writer to write the visualization frames to.
        pids (`list`): list of person ids.
    '''
    
    imgs = [None] * len(pcd)
    
    bg_img = bkg_image[0][0].clone().cpu().detach().numpy()
    bg_img = np.tile(bg_img[:,:,None], (1,1,3))
    bg_img = (bg_img * 255).astype('uint8')
    
    for i in range(len(pcd)):
        kps = pcd[i].cpu().detach().numpy()
        img = draw_kps(bg_img.copy(), kps, radius=1, thickness=1, color=(0, 255, 0))
        if pids is not None:
            imgs[int(pids[i])] = img
        else:
            imgs[i] = img
    
    curr_image = np.hstack(imgs)
    
    video_writer.write(curr_image[:,:,::-1])
