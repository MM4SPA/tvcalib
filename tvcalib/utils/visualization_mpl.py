import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.transforms.functional import to_pil_image
import seaborn as sns
import torch


from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.utils.objects_3d import Abstract3dModel

sns.set_style("whitegrid", {"axes.grid": False})


def _get_ax(b, t, batch_size, temporal_dim, axs):
    if batch_size == 1 and temporal_dim == 1:
        return axs
    elif batch_size > 1 and temporal_dim == 1:
        return axs[b]
    elif batch_size == 1 and temporal_dim > 1:
        return axs[t]
    else:
        return axs[b, t]


def visualize_project_points_px_batch(
    cam: SNProjectiveCamera,
    model3d: Abstract3dModel,
    img_delta_h=0.1,
    img_detla_w=0.2,
    figsize_scale=5,
    alpha=0.8,
    axs=None,
    true_bs=None,
    image_ids=None,
):

    if axs is None:
        _, axs = plt.subplots(
            nrows=cam.batch_dim,
            ncols=cam.temporal_dim,
            figsize=(cam.temporal_dim * figsize_scale, cam.batch_dim * figsize_scale),
            sharex=True,
            sharey=True,
        )

    cmap = {k: [c / 255.0 for c in v] for k, v in model3d.points_sampled_palette.items()}
    bs = cam.batch_dim
    if true_bs is not None:
        bs = true_bs

    for segment_name, points3d in model3d.points_sampled.items():
        with torch.no_grad():
            points_px = cam.project_point2pixel(points3d, lens_distortion=False).to(
                "cpu"
            )  # (N, 3) -> (B, T, N, 2)

        # TODO exclude "padded" points from visualization?

        for b in range(bs):
            for t in range(cam.temporal_dim):
                ax = _get_ax(b, t, bs, cam.temporal_dim, axs)
                title = cam.str_pan_tilt_roll_fl(b, t)
                if cam.psi is not None:  # concant lens dist. coeff.
                    title = title + " " + cam.str_lens_distortion_coeff(b)

                if image_ids is not None:
                    title = f"{image_ids[b][t]} - " + title

                ax.set_title(title, fontdict={"fontsize": 6})

                w, h = cam.image_width, cam.image_height
                # ax.add_patch(
                #     Rectangle(
                #         (0, 0),
                #         cam.image_width,
                #         cam.image_height,
                #         linewidth=1,
                #         edgecolor="k",
                #         facecolor="none",
                #     )
                # )
                ax.scatter(
                    points_px[b, t, :, 0],
                    points_px[b, t, :, 1],
                    s=0.1,  # , figsize_scale
                    facecolor=cmap[segment_name],
                    edgecolor=cmap[segment_name],
                    alpha=alpha,
                )
                ax.set_xlim([0 - w * img_detla_w, w + w * img_detla_w])
                ax.set_ylim([0 - h * img_delta_h, h + h * img_delta_h])
                ax.set_aspect("equal", "box")
                ax.invert_yaxis()

    for b in range(bs):
        if bs == 1 and cam.temporal_dim == 1:
            ax = axs
        elif bs > 1 and cam.temporal_dim == 1:
            ax = axs[b]
        elif bs == 1 and cam.temporal_dim > 1:
            ax = axs[0]
        else:
            ax = axs[b, 0]
        translation = (
            torch.stack([cam.phi_dict[k][b] for k in ["c_x", "c_y", "c_z"]], dim=-1)[0]
            .cpu()
            .tolist()
        )
        translation = ", ".join([f"{float(x):.1f}" for x in translation])
        ax.set_ylabel(f"t={translation}", size="small")

    fig = plt.gcf()
    fig.tight_layout()
    return axs


def visualize_annotated_points_px_batch(
    batch_img, points_line, points_circle, model3d, zorder=1000
):

    # print(points_line.shape)
    # print(points_line[:, :, 0].min(), points_line[:, :, 0].max())
    # print(points_line[:, :, 1].min(), points_line[:, :, 1].max())

    def _plot(ax, img, points_px_lines, points_px_circles, img_alpha=1.0):
        px_delta_w = 50
        px_delta_h = 25
        if img is not None:
            ax.imshow(to_pil_image(img), alpha=img_alpha)
        for s_l in range(points_px_lines.shape[-2]):
            ax.scatter(
                points_px_lines[0, s_l, ...],
                points_px_lines[1, s_l, ...],
                s=50,
                color=[c / 255 for c in model3d.line_palette[s_l]],
                alpha=0.3,
                facecolor="none",
                zorder=zorder,
            )
        for s_c in range(points_px_circles.shape[-2]):
            ax.scatter(
                points_px_circles[0, s_c, ...],
                points_px_circles[1, s_c, ...],
                s=50,
                alpha=0.3,
                facecolors="none",
                color=[c / 255 for c in model3d.circle_palette[s_c]],
                zorder=zorder,
            )

        for s_l in range(points_px_lines.shape[-2]):
            ax.scatter(
                points_px_lines[0, s_l, ...],
                points_px_lines[1, s_l, ...],
                s=5,
                marker=".",
                color="k",
                linewidths=0.5,
                zorder=zorder,
            )
        for s_c in range(points_px_circles.shape[-2]):
            ax.scatter(
                points_px_circles[0, s_c, ...],
                points_px_circles[1, s_c, ...],
                s=5,
                marker=".",
                color="k",
                linewidths=0.5,
                zorder=zorder,
            )
        # ax.set_xlim([-px_delta_w, img.shape[-1] + px_delta_w])
        # ax.set_ylim([-px_delta_h, img.shape[-2] + px_delta_h])
        ax.invert_yaxis()
        return ax

    B = points_line.shape[0]
    T = points_line.shape[1]

    figsize_scale = 5
    if B == 1 and T > 1:
        fig, axs = plt.subplots(
            nrows=B,
            ncols=T,
            figsize=(figsize_scale * T, figsize_scale * B),
            sharex=True,
            sharey=True,
        )
    elif B > 1 and T == 1:
        fig, axs = plt.subplots(
            nrows=B,
            ncols=T,
            figsize=(figsize_scale * T, figsize_scale * B / 1.77),
            sharex=True,
            sharey=True,
        )
    elif B > 1 and T > 1:
        fig, axs = plt.subplots(
            nrows=B,
            ncols=T,
            figsize=(figsize_scale * T, figsize_scale * B / 1.77),
            sharex=True,
            sharey=True,
        )
    else:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(figsize_scale, figsize_scale / 1.77),
        )

    for b in range(B):
        for t in range(T):
            if B == 1 and T == 1:
                ax = axs
            elif B > 1 and T == 1:
                ax = axs[b]
            elif B == 1 and T > 1:
                ax = axs[t]
            else:
                ax = axs[b, t]
            ax = _plot(
                ax,
                batch_img[b, t] if batch_img is not None else None,
                points_line[b, t],
                points_circle[b, t],
            )
    plt.tight_layout()
    return axs


def visualize_ndc_raster(x_dict, cam, object3d, xy_lim=True, pad_scale=0.3):

    # assume batch size == 1
    T = x_dict["lines__ndc_projected_selection_shuffled"].shape[1]

    fig, axs = plt.subplots(nrows=2, ncols=T, sharex=False, sharey=False, figsize=(10 * T, 10))
    cnmap_mpl = {k: [c / 255.0 for c in rgb] for k, rgb in object3d.points_sampled_palette.items()}

    with torch.no_grad():
        for b in range(T):
            ax = axs[0, b] if T > 1 else axs[0]
            ax.add_patch(
                Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor="k", facecolor="k", alpha=0.1)
            )

            for segment_name, points3d in object3d.points_sampled.items():
                projected_points_ndc = (
                    cam.project_point2ndc(points3d, lens_distortion=False).to("cpu").squeeze(0)
                )
                ax.scatter(
                    projected_points_ndc[b, :, 0],
                    projected_points_ndc[b, :, 1],
                    s=2,
                    color=cnmap_mpl[segment_name],
                )
            if xy_lim:
                ax.set_xlim([-1 - 2 * pad_scale, 1 + 2 * pad_scale])
                ax.set_ylim([-1 - 2 * pad_scale, 1 + 2 * pad_scale])
                ax.set_aspect("equal", "box")
            ax.invert_yaxis()
            ax.set_title(f"$fl_h=fl_v$={cam.intrinsics_ndc[b, 0, 0]:.2f} px")

        for b in range(T):
            ax = axs[1, b] if T > 1 else axs[1]
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    cam.image_width,
                    cam.image_height,
                    linewidth=2,
                    edgecolor="k",
                    facecolor="k",
                    alpha=0.1,
                )
            )
            for segment_name, points3d in object3d.points_sampled.items():
                projected_points_raster = (
                    cam.project_point2pixel(points3d, lens_distortion=False).to("cpu").squeeze(0)
                )
                ax.scatter(
                    projected_points_raster[b, :, 0],
                    projected_points_raster[b, :, 1],
                    s=2,
                    color=cnmap_mpl[segment_name],
                )
            if xy_lim:
                ax.set_xlim(
                    [0 - cam.image_width * pad_scale, cam.image_width + cam.image_width * pad_scale]
                )
                ax.set_ylim(
                    [
                        0 - cam.image_height * pad_scale,
                        cam.image_height + cam.image_height * pad_scale,
                    ]
                )
                ax.set_aspect("equal", "box")
            ax.invert_yaxis()
            ax.set_title(
                f"$fl_h$={cam.intrinsics_raster[b, 0, 0]:.2f} $px$ $fl_v$={cam.intrinsics_raster[b, 1, 1]:.2f} $px$"
            )

        # plt.tight_layout()
        fov_deg_formatted = [
            f"{deg:.1f}°" for deg in torch.rad2deg(cam.phi_dict_flat["aov"]).tolist()
        ]
        fov_deg_formatted = ", ".join(fov_deg_formatted)
        fig = plt.gcf()
        fig.suptitle(r"FOV" + f"= {fov_deg_formatted}", fontsize=12)

        points_ndc_lines_true = x_dict[
            "lines__ndc_projected_selection_shuffled"
        ]  # (B, T, 3, S_l, N_l)

        points_ndc_circles_true = x_dict[
            "circles__ndc_projected_selection_shuffled"
        ]  # (B, T, 3, S_l, N_l)

        points_px_lines_true = x_dict[
            "lines__px_projected_selection_shuffled"
        ]  # (B, T, 3, S_l, N_l)

        points_px_circles_true = x_dict[
            "circles__px_projected_selection_shuffled"
        ]  # (B, T, 3, S_l, N_l)

        cmap_lines = [[c / 255 for c in s] for s in object3d.line_palette]
        cmap_circles = [[c / 255 for c in s] for s in object3d.circle_palette]

        ppoints = [
            points_ndc_lines_true,
            points_ndc_circles_true,
            points_px_lines_true,
            points_px_circles_true,
        ]
        cmaps = [cmap_lines, cmap_circles, cmap_lines, cmap_circles]
        for i, (points, cmap) in enumerate(zip(ppoints, cmaps)):
            for t in range(T):
                if i == 0 or i == 1:
                    ax = axs[0, t] if T > 1 else axs[0]
                else:
                    ax = axs[1, t] if T > 1 else axs[1]
                for s in range(points.shape[-2]):
                    ax.scatter(
                        points[0, t, 0, s],
                        points[0, t, 1, s],
                        marker=7,  # CARETDOWN
                        s=50,
                        color=cmap[s],
                    )

    return fig, axs


def plot_per_step_loss(per_step_losses, image_ids=None, ylim=[0.0, 0.2]):

    batch_dim, temporal_dim, max_iter = per_step_losses.shape
    _, axs = plt.subplots(
        nrows=batch_dim,
        ncols=temporal_dim,
        sharex=True,
        sharey=True,
        figsize=(10, 2 * batch_dim),
    )

    for b in range(batch_dim):
        for t in range(temporal_dim):
            ax = _get_ax(b, t, batch_dim, temporal_dim, axs)
            ax.plot(per_step_losses[b, t])
            ax.grid(visible=True)
            ax.set_ylim(ylim)
            # ax.set_yscale("log")
            ax.set_xlim([0, max_iter])
            if image_ids:
                ax.set_title(str(image_ids[b][t]))

    plt.tight_layout()
    return ax


def plot_per_step_lr(per_step_lr):
    _, ax = plt.subplots(figsize=(8, 2))
    ax.plot(per_step_lr)
    ax.grid(visible=True)
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.tight_layout()
    return ax
