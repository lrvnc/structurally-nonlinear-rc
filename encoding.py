import torch
import numpy as np

#! Pytorch version
# def vectorized_basket(
#     tensor_inpt: torch.Tensor,
#     nbin: int,
#     min_val: float = -1.0,
#     max_val: float =  1.0,
#     clip: bool = False,
# ):
#     """
#     Vectorized basket encoding.

#     Params:
#         tensor_inpt: tensor[float32] of dimension (H, W) and values between [min_val, max_val]
#         nbin: int. number of bins to encode
#         min_val: float. Encoding lower limit
#         max_val: float. Encoding upper limit
#         clip: bool. Whether to clamp tensor_inpt to [min_val, max_val]
#     Returns:
#         tensor_bin: tensor[float32] of dimension (H, W * nbin)
#     """
#     H, W = tensor_inpt.shape
#     device = tensor_inpt.device
#     dtype  = tensor_inpt.dtype

#     assert min_val+0.25 <= tensor_inpt.min() and tensor_inpt.max() <= max_val-0.25, f'Basked encoding error: one or more values out of range ({tensor_inpt.min(), tensor_inpt.max()}).'

#     if clip:
#         tensor_inpt = tensor_inpt.clamp(min_val, max_val)

#     span = (max_val - min_val)

#     i = torch.arange(1, nbin + 1, device=device, dtype=dtype)

#     # Bins centered along [min_val, max_val]
#     c_i = min_val + ((2 * i - 1) / (2 * nbin)) * span  # shape: (nbin,)

#     s = ((2 * torch.floor(torch.tensor(nbin / 2.0, device=device, dtype=dtype)) - 1)
#          / (4 * nbin)) * span  # escalar

#     res = tensor_inpt.unsqueeze(-1)     # (H, W, 1)
#     c_i = c_i.view(1, 1, -1)            # (1, 1, nbin)

#     binarized = ((res >= c_i - s) & (res <= c_i + s)).to(dtype)  # (H, W, nbin)

#     # Reshape to (H, W * nbin)
#     tensor_bin = binarized.reshape(H, W * nbin)
#     return tensor_bin


def vectorized_basket(
    array_inpt: np.ndarray[np.float32],
    nbin: int,
    min_val: float = -1.0,
    max_val: float =  1.0,
    clip: bool = False,
) -> np.ndarray[np.float32]:
    """
    Vectorized basket encoding.

    Params:
        array_inpt: np.array[float32] of dimension (H, W) and values between [min_val, max_val]
        nbin: int. number of bins to encode
        min_val: float. Encoding lower limit
        max_val: float. Encoding upper limit
        clip: bool. Whether to clamp array_inpt to [min_val, max_val]
    Returns:
        array_bin: np.array[float32] of dimension (H, W * nbin)
    """
    H, W = array_inpt.shape

    assert min_val <= array_inpt.min() and array_inpt.max() <= max_val, f'Basked encoding error: one or more values out of range ({array_inpt.min(), array_inpt.max()}).'

    if clip:
        array_inpt = array_inpt.clamp(min_val, max_val)

    span = (max_val - min_val)

    i = np.arange(1, nbin + 1, dtype=np.float32)

    # Bins centered along [min_val, max_val]
    c_i = min_val + ((2 * i - 1) / (2 * nbin)) * span  # shape: (nbin,)

    s = ((2 * np.floor(nbin/2.0, dtype=np.float32) - 1) / (4 * nbin)) * span  # escalar

    res = np.expand_dims(array_inpt, axis=-1)     # (H, W, 1)
    c_i = c_i.reshape(1, 1, -1)            # (1, 1, nbin)

    binarized = ((res >= c_i - s) & (res <= c_i + s))  # (H, W, nbin)

    # Reshape to (H, W * nbin)
    array_bin = binarized.reshape(H, W * nbin)
    return array_bin.astype(np.float32)