import numpy as np
import scipy as scp


def corr2(img1, img2):
    assert img1.shape == img2.shape, "Images must be the same size"
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    num = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    den1 = np.sum((img1 - img1_mean)**2)
    den2 = np.sum((img2 - img2_mean)**2)
    den = np.sqrt(den1 * den2)
    r = num/den
    return r


def grid_downsample(img: np.ndarray, s: int, radius: int | None = None, agg: str = "mean", return_mask: bool = False) -> np.ndarray:
    """
    Downsample an image by sampling circular regions on a regular grid.

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (H, W) or (H, W, C).
    s : int
        Grid spacing (in pixels). Each grid cell is s×s.
    radius : int or None
        Radius (in pixels) of the circle inside each s×s cell.
        If None, defaults to s // 2.
    agg : {"mean", "median"}
        Aggregation to apply inside each circle.

    Returns
    -------
    np.ndarray
        Downsampled image of shape (H // s, W // s) or (H // s, W // s, C).
    """
    if s <= 0:
        raise ValueError("s must be a positive integer.")

    if img.ndim == 2:
        H, W = img.shape
        C = None
    elif img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError("img must be 2D (H, W) or 3D (H, W, C).")

    radius = radius if radius is not None else s // 2

    # Number of grid cells fully fitting inside the image
    n_rows = H // s
    n_cols = W // s

    if n_rows == 0 or n_cols == 0:
        raise ValueError("Grid spacing s is larger than the image dimensions.")
    
    img = img[:n_rows * s, :n_cols * s]

    # Precompute circular mask inside an s×s window
    yy, xx = np.ogrid[:s, :s]
    cy, cx = s // 2, s // 2  # center of the window
    circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    circle_mask = np.tile(circle_mask, reps=(n_rows, n_cols))
    downsampled_img = img[circle_mask]

    pixels_per_cell = circle_mask[:s, :s].sum()

    # Reshape into (n_cells, pixels_per_cell)
    downsampled_img = downsampled_img.reshape(-1, pixels_per_cell)

    # Apply aggregation
    if agg is None:
        pass
    elif agg == "mean":
        downsampled_img = downsampled_img.mean(axis=1)
    elif agg == "median":
        downsampled_img = np.median(downsampled_img, axis=1)
    else:
        raise ValueError("agg must be None, 'mean', or 'median'.")

    # Reshape
    if agg is None:
        if return_mask:
            return downsampled_img, circle_mask # each row one speckle grain
        else:
            return downsampled_img
    else:
        downsampled_img = downsampled_img.reshape(n_rows, n_cols) # each pixel one speckle grain
        if return_mask:
            return downsampled_img, circle_mask
        else:
            return downsampled_img


def gamma_k(k, d):
    k = np.abs(k)
    d_demean = d - d.mean()
    if k == 0:
        return (d_demean @ d_demean) / len(d_demean) # same as var(d)
    else:
        return (d_demean[k:] @ d_demean[:-k]) / len(d_demean)


def dm_test(d, h, alpha=0.05):

    d_bar = d.mean()
    T = len(d)
    assert h < T and h > 0

    if h == 1:
        den = gamma_k(k=0, d=d) / T # check denominator

    else:
        gamma_sum = 0
        for k in range(1, h): # h not included, goes until h-1
            gamma_sum += gamma_k(k=k, d=d)

        den = (gamma_k(k=0, d=d) + 2*gamma_sum) /  T # check denominator
    
    if den < 0:
        if np.isclose(den, 0, atol=1e-5):
            dm = np.inf
        else:
            raise ValueError(f'Denominator < 0 (den = {den})')
    else:
        dm = d_bar /  np.sqrt(den) # -> Std normal distribution

    dm_hln = np.sqrt((T + 1 - 2*h + h*(h-1)) / T) * dm # HLN correction # -> Student-t_{T-1} distribution

    # ===== Two tailed test =====
    # If I want to show that performances of model 1 and 2 are different, the hypothesis are:
    # H0: Model 1 and 2 have EQUAL accuracy
    # H1: Model 1 and 2 have DIFFERENT accuracy
    # We want to reject H0

    two_tailed_lower_t_crit = scp.stats.t.ppf(q=alpha/2, df=T-1)
    two_tailed_upper_t_crit = scp.stats.t.isf(q=alpha/2, df=T-1)
    # ===========================

    # ===== One tailed test =====
    # If I want to show that model 2 is better than model 1, the hypothesis are:
    # H0: e(model 2) >= e(model 1) --> d_t >= 0
    # H1: e(model 2) < e(model 1) --> d_t < 0
    # We want to reject H0

    # To do so, we craft d such as: d = MSE_2 - MSE_1 -> we want it to be negative, meaning the MSE of model 1 is bigger than model 2.
    # Once we compute the DM statistic, we do a ONE TAILED check: since we want d to be negative, the DM statistic must be smaller than a lower_t_crit to reject H0

    one_tailed_lower_t_crit = scp.stats.t.ppf(q=alpha, df=T-1)
    # ===========================

    result = {
        'dm': dm,
        'p_value': scp.stats.norm.cdf(dm),
        'dm_hln': dm_hln,
        't_value': scp.stats.t.cdf(dm_hln, df=T-1),
        'reject_H0_two_tailed': dm_hln < two_tailed_lower_t_crit or dm_hln > two_tailed_upper_t_crit,
        'reject_H0_one_tailed': dm_hln < one_tailed_lower_t_crit # if True, model 2 is better.
    }

    return result


def validate_data_split_params(params):
    train = params.get('train_phase', {})
    test = params.get('test_phase', {})

    # --- train_phase ---
    forget = train.get('forget')
    n_train = train.get('number_train_timesteps')

    if forget is None or forget < 0:
        raise ValueError("forget must be >= 0")

    if n_train is None or n_train <= forget:
        raise ValueError("number_train_timesteps must be > forget")

    # --- test_phase ---
    pred_h = test.get('prediction_horizon')
    if pred_h is None or pred_h < 1:
        raise ValueError("prediction_horizon must be >= 1")

    n_origins = test.get('number_forecast_origins')
    if n_origins is None or n_origins < 1:
        raise ValueError("number_forecast_origins must be >= 1")

    spacing = test.get('forecast_origin_spacing')
    if spacing is None or spacing < 0:
        raise ValueError("forecast_origin_spacing must be >= 0")

    if n_origins == 1 and spacing != 0:
        raise ValueError("forecast_origin_spacing can be > 0 only if number_forecast_origins > 1")

    warmup = test.get('number_warmup_timesteps')
    if warmup is None or warmup < 0:
        raise ValueError("number_warmup_timesteps must be >= 0")