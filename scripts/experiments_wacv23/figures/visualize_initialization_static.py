#%%
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import tvcalib.utils.visualization_mpl_min as viz 
from tvcalib.cam_modules import CameraParameterWLensDistDictZScore, SNProjectiveCamera
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSN
import tvcalib.cam_distr.tv_main_center
import tvcalib.cam_distr.tv_main_left
import tvcalib.cam_distr.tv_main_right


output_dir = Path("data/visualization")
output_dir.mkdir(exist_ok=True, parents=True)

base_field = SoccerPitchSN()
object3d = SoccerPitchLineCircleSegments(base_field=base_field)

#%%
dist_distr = None
candidates = {
    "left": tvcalib.cam_distr.tv_main_left.get_cam_distr(1.96, 1, 1), 
    "center": tvcalib.cam_distr.tv_main_center.get_cam_distr(1.96, 1, 1), 
    "right": tvcalib.cam_distr.tv_main_right.get_cam_distr(1.96, 1, 1),
}
for candidate_name, cam_distr in candidates.items():
    with torch.no_grad():
        cam_param_dict = CameraParameterWLensDistDictZScore(cam_distr, dist_distr)
        phi_hat, psi_hat = cam_param_dict()
        print(cam)
        cam = SNProjectiveCamera(
            phi_hat,
            psi_hat,
            (960 / 2, 540 / 2),
            960,
            540,
            nan_check=False,
    )

    _, ax = viz.init_figure(960, 540, img_delta_w=0.0, img_delta_h=0.0)
    ax = viz.draw_reprojection(ax, object3d, cam, dist_circles=0.25, kwargs={"linewidth": 8})

    dpi = 50
    plt.savefig(output_dir / f"{candidate_name}.pdf", dpi=dpi)
    plt.savefig(output_dir / f"{candidate_name}.svg", dpi=dpi)
    plt.savefig(output_dir / f"{candidate_name}.png", dpi=dpi)
# %%
