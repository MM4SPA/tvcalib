import torch
import numpy as np
import kornia
import scipy.ndimage
from functools import partial

from SoccerNet.Evaluation.utils_calibration import Camera


torch.set_printoptions(precision=3, sci_mode=False)
np.set_printoptions(precision=3, suppress=True)
batch_size = 1


def iou_whole(h_pred, h_true, scaling_factor=4, eps=1e-6):

    fake_template = torch.ones([1, 1, int(68 * scaling_factor), int(105 * scaling_factor)])
    scaling_mat = torch.eye(3).repeat(1, 1, 1)
    scaling_mat[:, 0, 0] = scaling_mat[:, 1, 1] = scaling_factor
    warp = partial(kornia.geometry.transform.homography_warp, normalized_coordinates=True, normalized_homography=True)

    target_mask = warp(fake_template.clone(), scaling_mat, dsize=fake_template.shape[-2:])

    normalize_h = partial(kornia.geometry.normalize_homography, dsize_src=(68, 105), dsize_dst=(68, 105))

    h_pred = normalize_h(h_pred.inverse())
    h_true = normalize_h(h_true.inverse())

    mapping_mat = h_true.inverse() @ h_pred @ scaling_mat
    output_mask = warp(fake_template.clone(), mapping_mat, dsize=fake_template.shape[-2:])

    # iou calculation
    output_mask = (output_mask >= 0.5).float()
    target_mask = (target_mask >= 0.5).float()
    intersection_mask = output_mask * target_mask
    output = output_mask.sum(dim=[1, 2, 3])
    target = target_mask.sum(dim=[1, 2, 3])
    intersection = intersection_mask.sum(dim=[1, 2, 3])
    union = output + target - intersection
    iou = intersection / (union + eps)

    img_composite = torch.zeros((1, 3, int(68*scaling_factor), int(105*scaling_factor)), dtype=torch.uint8)
    img_composite[:, 0] = target_mask[:, 0]
    img_composite[:, 2] = output_mask[:, 0]
    img_composite[(img_composite.sum(dim=1) == 2.0).unsqueeze(1).repeat(1, 3, 1, 1)] = 1.0
    img_composite = (img_composite * 255.0).to(torch.uint8)
    return iou, img_composite
    

def iou_part(h_pred, h_true, image_height: int, image_width: int, eps=1e-6):

    def _warp_ones2_sn_template(H, image_height: int, image_width: int):
        # H that maps from image to template
        # translate center of the homography matrix to image origin (upper left)
        T = torch.eye(3).unsqueeze(0)
        T[:, 0, -1] = 105 / 2
        T[:, 1, -1] = 68 / 2
        H = T @ H @ T
        warped_top = kornia.geometry.transform.homography_warp(
            torch.ones(1, 1, image_height, image_width),
            H,
            dsize=(68, 105),
            normalized_homography=False,
            normalized_coordinates=False,
        )
        warped_top = torch.from_numpy(scipy.ndimage.binary_fill_holes(warped_top))
        return warped_top.to(torch.uint8)

    output_mask = _warp_ones2_sn_template(h_pred, image_height, image_width) # (B, 1, 68, 105)
    target_mask = _warp_ones2_sn_template(h_true, image_height, image_width) # (B, 1, 68, 105)

    output_mask[output_mask > 0] = 1
    target_mask[target_mask > 0] = 1

    intersection_mask = output_mask * target_mask
    output = output_mask.sum(dim=[1, 2, 3])
    target = target_mask.sum(dim=[1, 2, 3])
    intersection = intersection_mask.sum(dim=[1, 2, 3])
    union = output + target - intersection
    iou = intersection / (union + eps)

    img_composite = torch.zeros((1, 3, 68, 105), dtype=torch.uint8)
    img_composite[:, 0] = target_mask[:, 0]
    img_composite[:, 2] = output_mask[:, 0]
    img_composite[(img_composite.sum(dim=1) == 2.0).unsqueeze(1).repeat(1, 3, 1, 1)] = 1.0
    img_composite = (img_composite * 255.0).to(torch.uint8)
    
    return iou, img_composite


def get_sn_homography(cam_params: dict):
    cam = Camera(iwidth=1280, iheight=720)
    cam.from_json_parameters(cam_params)
    It = np.eye(4)[:-1]
    It[:, -1] = -cam.position  # (3, 4)
    P = cam.calibration @ cam.rotation @ It  #  # (3, 4)
    H = P[:, [0, 1, 3]]  # (3, 3)
    H = torch.from_numpy(H).float()
    H = H.unsqueeze(0).repeat(batch_size, 1, 1)
    H = H / H[:, -1, -1]  # normalize homography
    return H


def get_wc14_homography(file_cam_params):
    # *.mat
    H = torch.from_numpy(np.loadtxt(file_cam_params)).float()
    return H.unsqueeze(0).repeat(batch_size, 1, 1)
