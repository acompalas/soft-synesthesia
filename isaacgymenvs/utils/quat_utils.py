import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Drop-in for pytorch3d.transforms.quaternion_to_matrix. Input (..., 4), real part first (wxyz)."""
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(list(quaternions.shape[:-1]) + [3, 3])


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Drop-in for pytorch3d.transforms.quaternion_apply. Rotates point (..., 3) by quaternion (..., 4), wxyz."""
    rot = quaternion_to_matrix(quaternion)
    return torch.matmul(rot, point.unsqueeze(-1)).squeeze(-1)
