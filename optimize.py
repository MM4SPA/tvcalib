from argparse import ArgumentParser
import json
import random
from pathlib import Path
from time import time
from datetime import datetime

import torch
import matplotlib.pyplot as plt
from tvcalib.cam_modules import SNProjectiveCamera

from tvcalib.datasets.sn_calib import FixedInputSizeDataset, custom_list_collate
from tvcalib.tvcalib_module import TVCalibOptim
from tvcalib.utils.objects_3d import (
    SoccerPitchLineCircleSegments,
    SoccerPitchSN,
    SoccerPitchSNCircleCentralSplit,
)
import tvcalib.utils.io as io
from tvcalib.utils.visualization_mpl import (
    plot_per_step_loss,
    plot_per_step_lr,
    visualize_annotated_points_px_batch,
    visualize_project_points_px_batch,
)


torch.set_printoptions(precision=3, sci_mode=False)
torch.manual_seed(0)
random.seed(0)


# TODO: default arg from dict when using ipython during dev
args = ArgumentParser()
args.add_argument("--hparams", type=Path)
args.add_argument("--log_per_step", action="store_true")
args.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
args.add_argument("--visualize_results", action="store_true")
args.add_argument("--output_dir", type=Path, default="experiments")
args.add_argument("--exp_timestmap", action="store_true")
args.add_argument("--overwrite_init_cam_distr", type=str)
args = args.parse_args()


HPARAMS_FILE = args.hparams
hparams_prefix = HPARAMS_FILE.stem
LOG_PER_STEP = args.log_per_step
VISUALIZE_RESULTS = args.visualize_results
DEVICE = args.device
OUTPUT_DIR = args.output_dir

with open(HPARAMS_FILE) as fw:
    hparams = json.load(fw)

init_cam_distr = hparams["dataset"]["filter_cam_type"]
# TODO generic import
if args.overwrite_init_cam_distr is not None:
    init_cam_distr = args.overwrite_init_cam_distr

if init_cam_distr == "Main camera left":
    from tvcalib.cam_distr_configs.tv_main_left import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main camera right":
    from tvcalib.cam_distr_configs.tv_main_right import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main behind the goal":
    from tvcalib.cam_distr_configs.tv_main_behind import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main camera center":
    from tvcalib.cam_distr_configs.tv_main_center import get_cam_distr, get_dist_distr
elif init_cam_distr == "Main tribune":
    from tvcalib.cam_distr_configs.tv_main_tribune import get_cam_distr, get_dist_distr

    hparams["dataset"]["filter_cam_type"] = False
else:
    from tvcalib.cam_distr_configs.tv_main_center import get_cam_distr, get_dist_distr


distr_lens_disto = None
distr_cam = get_cam_distr(hparams["sigma_scale"], hparams["batch_dim"], hparams["temporal_dim"])
if hparams["lens_distortion"] == True:
    distr_lens_disto = get_dist_distr(hparams["batch_dim"], hparams["temporal_dim"])
hparams["distr_cam"] = distr_cam
hparams["distr_lens_disto"] = distr_lens_disto


output_dir = OUTPUT_DIR / hparams_prefix
if args.exp_timestmap:
    output_dir = output_dir / datetime.now().strftime("%y%m%d-%H%M")

output_dir.mkdir(exist_ok=True, parents=True)
print("output directory", output_dir)

if (
    "split_circle_central" in hparams["dataset"]
    and hparams["dataset"]["split_circle_central"] == True
):
    base_field = SoccerPitchSNCircleCentralSplit()
else:
    base_field = SoccerPitchSN()
object3d = SoccerPitchLineCircleSegments(device=DEVICE, base_field=base_field)
print(base_field.__class__, object3d.__class__)

print("Init Dataset")
dataset = FixedInputSizeDataset(
    model3d=object3d,
    image_width=hparams["image_width"],
    image_height=hparams["image_height"],
    constant_cam_position=hparams["temporal_dim"],
    **hparams["dataset"],
)
print(dataset.df_match_info.head(5))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=hparams["batch_dim"],
    num_workers=4,
    shuffle=False,
    collate_fn=custom_list_collate,
)

print("Init TVCalibOptim")
model = TVCalibOptim(
    object3d,
    distr_cam,
    distr_lens_disto,
    (hparams["image_height"], hparams["image_width"]),
    hparams["optim_steps"],
    DEVICE,
    log_per_step=LOG_PER_STEP,
)
hparams["TVCalibOptim"] = model.hparams
print(output_dir / "hparams.yml")
io.write_yaml(hparams, output_dir / "hparams.yml")


for batch_idx, x_dict in enumerate(dataloader):

    print(f"{batch_idx}/{len(dataloader)-1}")
    points_line = x_dict["lines__px_projected_selection_shuffled"].clone().detach()
    points_circle = x_dict["circles__px_projected_selection_shuffled"].clone().detach()
    batch_size = points_line.shape[0]

    fout_prefix = f"batch_{batch_idx}"

    start = time()
    per_sample_loss, cam, per_step_info = model.self_optim_batch(x_dict)

    output_dict = {
        "image_ids": x_dict["image_id"],
        "camera": cam.get_parameters(batch_size),
        "time_s": time() - start,
        **per_sample_loss,
        "meta": x_dict["meta"],
    }
    if LOG_PER_STEP:
        output_dict["per_step_lr"] = per_step_info["lr"].squeeze(-1)  # (optim_steps,)
        output_dict["per_step_loss"] = per_step_info["loss"]  # (B, T, optim_steps)

    print(output_dir / f"{fout_prefix}.pt")

    def detach_fn(x_dict):
        with torch.no_grad():
            for k in x_dict.keys():
                if isinstance(x_dict[k], torch.Tensor):
                    x_dict[k] = x_dict[k].detach().cpu()
                elif isinstance(x_dict[k], dict):
                    x_dict[k] = detach_fn(x_dict[k])
        return x_dict

    torch.save(detach_fn(output_dict), output_dir / f"{fout_prefix}.pt")

    time_start_visualization = time()

    if VISUALIZE_RESULTS:
        with torch.no_grad():
            images = x_dict["image"]
            if distr_lens_disto is not None:
                # we visualize annotated points and image after undistortion
                # as the objective function calculates distances at undistorted NDC space
                images = cam.undistort_images(images)
                points_line = (
                    SNProjectiveCamera.static_undistort_points(points_line.to(DEVICE), cam)
                    .detach()
                    .cpu()
                )
                points_circle = (
                    SNProjectiveCamera.static_undistort_points(points_circle.to(DEVICE), cam)
                    .detach()
                    .cpu()
                )
            images = (images * 255.0).type(torch.uint8).detach().cpu()
            # object (pitch) overlay on padded image

            # TODO: slow for larger batch sizes (> 16)
            # TODO: solution: per sample plot with multiprocessing.pool or shift to eval/analyize script
            axs = visualize_annotated_points_px_batch(images, points_line, points_circle, object3d)
            _ = visualize_project_points_px_batch(
                cam, object3d, axs=axs, true_bs=batch_size, image_ids=output_dict["image_ids"]
            )
            print(output_dir / f"{fout_prefix}.pdf")
            plt.savefig(output_dir / f"{fout_prefix}.pdf")

            if LOG_PER_STEP:
                _ = plot_per_step_loss(per_step_info["loss"].cpu(), output_dict["image_ids"])
                print(output_dir / f"{fout_prefix}_loss.pdf")
                plt.savefig(output_dir / f"{fout_prefix}_loss.pdf")
                _ = plot_per_step_lr(per_step_info["lr"].cpu())
                print(output_dir / f"{fout_prefix}_lr.pdf")
                plt.savefig(output_dir / f"{fout_prefix}_lr.pdf")
            print("Time for visualizations [s]:", f"{time() - time_start_visualization:.1f}")
        plt.close("all")
