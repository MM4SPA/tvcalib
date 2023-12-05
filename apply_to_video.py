from collections import defaultdict
from functools import partial

import numpy as np
import torch
# import matplotlib
# matplotlib.use("Qt5Agg")
# matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.ioff()

from tqdm.auto import tqdm
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import SoccerPitch

from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate, split_circle_central
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output
from tvcalib.utils import visualization_mpl_min as viz

import imageio
from skimage import img_as_ubyte
from skimage.transform import resize
from scipy.signal import savgol_filter


# def show(img):
#     plt.figure()
#     plt.imshow(img)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()


# read input video
# frame_raw_list stores frames of original resolutions of the video
# frame_list stores frames resized to (256, 455) so that the model can receive it.
input_video_name = 'demo_video_trim.mp4'

reader = imageio.get_reader(input_video_name)
fps = reader.get_meta_data()['fps']
frame_list = []
frame_raw_list = []
for im in tqdm(reader, desc="reading video", total=reader.count_frames()):
    frame_list.append(resize(im, (256, 455), order=3))
    frame_raw_list.append(im/256)
reader.close()
frame_shape = frame_list[0].shape[:2]
frame_raw_shape = frame_raw_list[0].shape[:2]


# detecting keypoints from the frames
device = "cuda"

object3d = SoccerPitchLineCircleSegments(
    device=device, base_field=SoccerPitchSNCircleCentralSplit()
)
object3dcpu = SoccerPitchLineCircleSegments(
    device="cpu", base_field=SoccerPitchSNCircleCentralSplit()
)

lines_palette = [0, 0, 0]
for line_class in SoccerPitch.lines_classes:
    lines_palette.extend(SoccerPitch.palette[line_class])

fn_generate_class_synthesis = partial(generate_class_synthesis, radius=4)
fn_get_line_extremities = partial(get_line_extremities, maxdist=30, width=frame_shape[1], height=frame_shape[0],
                                  num_points_lines=4, num_points_circles=8)


model_seg = InferenceSegmentationModel("data\\segment_localization\\train_59.pt", device)


keypoints_raw_li = []
for data in tqdm(frame_list, desc="extracting keypoints"):
    data = torch.FloatTensor(np.transpose(data, (2, 0, 1))).to(device)
    with torch.no_grad():
        sem_lines = model_seg.inference(data.unsqueeze(0))
    sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

    skeletons = fn_generate_class_synthesis(np.squeeze(sem_lines, axis=0))
    keypoints_raw = fn_get_line_extremities(skeletons)

    keypoints_raw_li.append(keypoints_raw)


# Extract camera variables base on the key points obtained above
lens_dist = False

batch_size = 256
optim_steps = 2000

model_calib = TVCalibModule(
    object3d,
    get_cam_distr(1.96, batch_dim=batch_size, temporal_dim=1),
    get_dist_distr(batch_dim=batch_size, temporal_dim=1) if lens_dist else None,
    frame_shape,
    optim_steps,
    device,
    log_per_step=False,
    tqdm_kwqargs=None,
)


dataset_calib = InferenceDatasetCalibration(keypoints_raw_li, frame_shape[1], frame_shape[0], object3d)
dataloader_calib = torch.utils.data.DataLoader(dataset_calib, batch_size, collate_fn=custom_list_collate)


per_sample_output = defaultdict(list)
per_sample_output["image_id"] = [[x] for x in range(len(frame_list))]
for x_dict in dataloader_calib:
    _batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

    points_line = x_dict["lines__px_projected_selection_shuffled"]
    points_circle = x_dict["circles__px_projected_selection_shuffled"]

    per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)
    output_dict = tensor2list(detach_dict({**cam.get_parameters(_batch_size), **per_sample_loss}))

    output_dict["points_line"] = points_line
    output_dict["points_circle"] = points_circle
    for k in output_dict.keys():
        per_sample_output[k].extend(output_dict[k])


df = pd.DataFrame.from_dict(per_sample_output)

df = df.explode(column=[k for k, v in per_sample_output.items() if isinstance(v, list)])
df.set_index("image_id", inplace=True, drop=False)


# Smoothing using the savgol filter
window = 31
df.aov_radian = savgol_filter(df.aov_radian, window, 3)
df.pan_degrees = savgol_filter(df.pan_degrees, window, 3)
df.roll_degrees = savgol_filter(df.roll_degrees, window, 3)
df.tilt_degrees = savgol_filter(df.tilt_degrees, window, 3)
df.position_meters = pd.Series(savgol_filter(np.array(df.position_meters.to_list()), window, 3, axis=0).tolist())


plt.ioff()
result = []
for i in tqdm(range(len(frame_list)), desc="generating final video"):
    sample = df.iloc[i]

    image_raw = torch.tensor(np.transpose(frame_raw_list[i], (2, 0, 1)))

    cam = get_camera_from_per_sample_output(sample, lens_dist)

    fig, ax = viz.init_figure(frame_raw_shape[1], frame_raw_shape[0])
    ax = viz.draw_image(ax, image_raw)
    ax = viz.draw_reprojection(ax, object3dcpu, cam, ratio_width=frame_raw_shape[1]/frame_shape[1],
                               ratio_height=frame_raw_shape[1]/frame_shape[1])

    fig.canvas.draw()
    result.append(np.array(fig.canvas.renderer._renderer))
    plt.close()

imageio.mimsave("result.mp4", [img_as_ubyte(p) for p in result], fps=fps)
