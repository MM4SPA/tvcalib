from typing import Optional, Union
import torch
from kornia.geometry.conversions import convert_points_from_homogeneous


class LineCollection:
    def __init__(
        self,
        support: torch.tensor,
        direction_norm: torch.tensor,
        direction: Optional[torch.tensor] = None,
    ):
        """Wrapper class to represent lines by support and direction vectors.

        Args:
            support (torch.tensor): with shape (*, {2,3})
            direction_norm (torch.tensor): with shape (*, {2,3})
            direction (Optional[torch.tensor], optional): Unnormalized direction vector. Defaults to None.
        """
        self.support = support
        self.direction_norm = direction_norm
        self.direction = direction

    def __copy__(self):
        return LineCollection(
            self.support.clone(),
            self.direction_norm.clone(),
            self.direction.clone() if self.direction is not None else None,
        )

    def copy(self):
        return self.__copy__()

    def shape(self):
        return f"support={self.support.shape} direction_norm={self.direction_norm.shape} direction={self.direction.shape if self.direction else None}"

    def __repr__(self) -> str:
        return f"{self.__class__} " + self.shape()


def distance_line_pointcloud_3d(
    e1: torch.Tensor,
    r1: torch.Tensor,
    pc: torch.Tensor,
    reduce: Union[None, str] = None,
    eps=1e-6,
    nan_check=False,
) -> torch.Tensor:
    """
    Line to point cloud distance with arbitrary leading dimensions.

    TODO. if cross = (0.0.0) -> distance=0 otherwise NaNs are returned

    https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    Args:
        e1 (torch.Tensor): direction vector of shape (*, B, 1, 3)
        r1 (torch.Tensor): support vector of shape (*, B, 1, 3)
        pc (torch.Tensor): point cloud of shape (*, B, A, 3)
        reduce (Union[None, str]): reduce distance for all points to one using 'mean' or 'min'
    Returns:
        distance of an infinite line to given points, (*, B, ) using reduce='mean' or reduce='min' or (*, B, A) if reduce=False
    """

    num_points = pc.shape[-2]
    _sub = r1 - pc  # (*, B, A, 3)

    # assert torch.allclose(e1.repeat(1, num_points, 1), e1.repeat_interleave(num_points, dim=1))
    cross = torch.cross(e1.repeat_interleave(num_points, dim=-2), _sub, dim=-1)  # (*, B, A, 3)
    # cross = torch.cross(e1.repeat(1, num_points, 1), _sub, dim=-1)  # (*, B, A, 3)

    e1_norm = torch.linalg.norm(e1, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1)

    d = cross_norm / e1_norm
    if reduce == "mean":
        return d.mean(dim=-1)  # (*, B, )
    elif reduce == "min":
        return d.min(dim=-1)[0]  # (*, B, )

    # # no reduction
    # if nan_check:
    #     if torch.isnan(d).any().item():
    #         raise RuntimeWarning("NaNs in distance_line_pointcloud_3d")

    return d  # (B, A)


def distance_point_pointcloud(points: torch.Tensor, pointcloud: torch.Tensor) -> torch.Tensor:
    """Batched version for point-pointcloud distance calculation
    Args:
        points (torch.Tensor): N points in homogenous coordinates; shape (B, T, 3, S, N)
        pointcloud (torch.Tensor): N_star points for each pointcloud; shape (B, T, S, N_star, 2)

    Returns:
        torch.Tensor: Minimum distance for each point N to pointcloud; shape (B, T, 1, S, N)
    """

    batch_size, T, _, S, N = points.shape
    batch_size, T, S, N_star, _ = pointcloud.shape

    pointcloud = pointcloud.reshape(batch_size * T * S, N_star, 2)

    points = convert_points_from_homogeneous(
        points.permute(0, 1, 3, 4, 2).reshape(batch_size * T * S, N, 3)
    )

    # cdist signature: (B, P, M), (B, R, M) -> (B, P, R)
    distances = torch.cdist(points, pointcloud, p=2)  # (B*T*S, N, N_star)

    distances = distances.view(batch_size, T, S, N, N_star)
    distances = distances.unsqueeze(-4)

    # distance to nearest point from point cloud (batch_size, T, 1, S, N, N_star)
    distances = distances.min(dim=-1)[0]
    return distances


def distance_finite_line_pointcloud(lp1: torch.Tensor, lp2: torch.Tensor, pc: torch.Tensor):
    """Given a batch (of size '*') of B lines and a batch of A points (point cloud), computes the distance between each finite line and all points.

    Todo: Test!

    https://monkeyproofsolutions.nl/wordpress/how-to-calculate-the-shortest-distance-between-a-point-and-a-line/

    e = lp2 - lp1
    t0 = dot(pc - lp1, e) / dot(e, e)

    # points on line from orthogonal projection
    pnt_intersect = lp1 + t0 * e


    distance for finite lines:
    case 0: 0 < t0 < 1 -> norm(pc - pnt_intersect)
    case 1: t0 <= 0 -> norm(pc - lp1)
    case 2: t0 >= 1 -> norm(pc - lp2)

    distance for infinite lines: always case 0

    Args:
        lp1 (torch.tensor): start point of line of shape (batch_dim, temporal_dim, B, 1, D)
        lp2 (torch.tensor): end point of line of shape (batch_dim, temporal_dim, B, 1, D)
        pc (torch.tensor): point cloud of shape (batch_dim, temporal_dim, B, A, D)
    Returns:
        distance of an finite line to given point clouds (return tensor of shape (B, A))
    """

    batch_dim, temporal_dim, S, _, D = lp1.shape
    lp1 = lp1.view(batch_dim * temporal_dim, S, 1, D)
    lp2 = lp2.view(batch_dim * temporal_dim, S, 1, D)

    N = pc.shape[-2]
    pc = pc.view(batch_dim * temporal_dim, S, N, D)

    lp12 = lp2 - lp1
    BS, B, _, D = lp12.shape  # (BS, B, 1, D)
    A = pc.shape[-2]  # pc of shape (BS, 1, A, D)

    lp1_rep = lp1.repeat(1, 1, A, 1)
    lp2_rep = lp2.repeat(1, 1, A, 1)

    lp1pc = pc - lp1_rep  # (BS, B, A, D)
    t0 = torch.bmm(
        lp1pc.reshape(BS * B * A, 1, D), lp2_rep.reshape(BS * B * A, D, 1)
    )  # batch-wise dot product

    t0 = t0.view(BS, B, A, 1)

    dot_lp12 = torch.bmm(lp12.view(BS * B, 1, D), lp12.view(BS * B, 1, D).transpose(-1, -2)).view(
        BS, B, 1, 1
    )

    t0 = t0 / dot_lp12  # (BS, B, A, 1)

    # case 0 is default
    pnt_intersect = lp1_rep + t0 * lp12.repeat(1, 1, A, 1)  # (BS, B, A, D)

    d = torch.linalg.norm(pc - pnt_intersect, dim=-1)  # (BS, B, A, 3) -> (BS, B, A)
    # flatten for easy mask replace
    d = d.view(BS * B * A)
    t0 = t0.view(BS * B * A)

    # replace
    d_case1 = torch.linalg.norm(pc - lp1.repeat(1, 1, A, 1), dim=-1).view(BS * B * A)
    d_case2 = torch.linalg.norm(pc - lp2.repeat(1, 1, A, 1), dim=-1).view(BS * B * A)

    mask_case1 = t0 < 0
    d[mask_case1] = d_case1[mask_case1]
    mask_case2 = t0 > 1
    d[mask_case2] = d_case2[mask_case2]
    d = d.view(BS, B, A)

    d = d.view(batch_dim, temporal_dim, B, A)
    return d
