import torch
import numpy as np
import scipy as scp
import torch.nn.functional as F


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



def downsample_tensor(img: np.ndarray, k: int, s: int, pooling: str = None):

    if pooling == 'max':
        img_downsampled = F.max_pool2d(img.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=s, ceil_mode=False).squeeze()
    elif pooling == 'avg':
        img_downsampled = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=s, ceil_mode=False).squeeze()
    elif pooling == 'onepx':
        img_downsampled = img[::s, ::s]
    elif pooling == 'simple':
        h, w = img.shape
        mask = torch.ones((k, k))
        top, bottom = int(np.floor(s/4)), int(np.ceil(s/4))
        left, right = int(np.floor(s/4)), int(np.ceil(s/4))
        mask = F.pad(mask, pad=(left,right,top,bottom), value=0)
        mask = mask.repeat(int(np.ceil(h/s)),int(np.ceil(w/s))).to(torch.bool)
        mask = mask[:h, :w]

        img_downsampled = img[mask]

        h_down = mask[:, top].sum()
        w_down = mask[left, :].sum()

        img_downsampled = img_downsampled.reshape(h_down, w_down)
    else:
        img_downsampled = img # Do nothing :)
    return img_downsampled


def mse(pred, tgt):
    return np.mean((tgt - pred)**2)


def mae(pred, tgt):
    return np.mean(np.abs(tgt - pred))


def mape(pred, tgt, eps=1e-7):
    return 100 * np.mean(np.abs((tgt - pred)/(tgt + eps)))


def smape(pred, tgt, eps=1e-7):
    return 100 * np.mean(np.abs(pred - tgt) / ((np.abs(tgt) + np.abs(pred)) / 2 + eps))


class CausalSmoother:
    """
    Causal Smoother class, with arbitrary kernel.

    The kernel must be 1D, shape (K,), only past→future.
    Convention:
        kernel[0] multiply x_t        (actual timepoint)
        kernel[1] multiply x_{t-1}    (1 step in the past)
        ...
        kernel[K-1] multiply x_{t-K+1)}

    Example: causal Moving Average (MA), window=5:
        kernel = np.ones(5)

    Example: 'Gaussian kernel', window=15
        k = np.arange(0, 15)
        kernel = np.exp(-0.5 * (k/sigma)**2)
    """

    def __init__(self, kernel: np.ndarray):
        kernel = np.asarray(kernel, dtype=np.float32)
        assert kernel.ndim == 1, "Kernel must be 1D (K,)"
        assert kernel.size >= 1, "Kernel can't be empty"
        assert np.all(kernel >= 0), "Kernel must be >= for smoothening"

        # Normalize to 1 (prevent arbitrary gain)
        self.kernel = kernel / kernel.sum()

        # Internal buffer (when online smoothening)
        self._buffer = []  # most recent in position 0

    def reset(self):
        self._buffer = []

    def step(self, x_t: np.ndarray) -> np.ndarray:
        """
        Updates the filter with the new sample x_t and returns y_t smoothed.

        x_t: shape (D,) or (D,1)
        returns: shape (D,)
        """
        x_t = np.asarray(x_t, dtype=np.float32).reshape(-1)

        # Append sample in the beginning of the buffer
        self._buffer.insert(0, x_t.copy())

        # Cut the buffer in the size of the kernel
        if len(self._buffer) > len(self.kernel):
            self._buffer = self._buffer[:len(self.kernel)]

        # Apply the kernel weights
        y_t = np.zeros_like(x_t, dtype=np.float32)
        for k, w in enumerate(self.kernel[:len(self._buffer)]):
            y_t += w * self._buffer[k]

        return y_t  # (D,)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the causal filter in the entire series.

        X: shape (T, D)
        returns: shape (T, D)
        """
        X = np.asarray(X, dtype=np.float32)
        T, D = X.shape
        Y = np.zeros_like(X, dtype=np.float32)

        self.reset()
        for t in range(T):
            Y[t] = self.step(X[t])
        return Y


def amp_spectrum(signal: np.ndarray[np.float32], dt: float):
    Xf = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=dt)

    mag = (2.0/len(signal)) * np.abs(Xf)
    mag[0] = mag[0]/2.0

    return freqs, mag


def step_l2_distance(states: np.ndarray) -> np.ndarray:
    """
    Calculates d[t] = || x_t - x_{t-1} ||_2 para t=1..T-1.
    states: (n, T)
    returns: dist array (T-1,)
    """
    # diffs: shape (n, T-1)
    diffs = states[:, 1:] - states[:, :-1]

    # L2 norm along the features' axis (n)
    # sqrt(sum_i diff_i^2)
    dist = np.sqrt(np.sum(diffs**2, axis=0, dtype=np.float64)).astype(np.float32)

    return dist


def max_eig_power(A, n_iter=200, tol=1e-6):
    # garante vetor complexo
    n = A.shape[0]
    v = (np.random.randn(n) + 1j*np.random.randn(n)).astype(np.complex64)
    v /= np.linalg.norm(v)

    lam_old = 0.0 + 0.0j
    for _ in range(n_iter):
        w = A @ v
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            return 0.0
        v = w / norm_w
        # usa o Rayleigh sobre o w já atualizado
        lam = v.conj().T @ (A @ v)
        if np.abs(lam - lam_old) < tol * np.abs(lam):
            break
        lam_old = lam
    return lam


def max_singular(A, n_iter=50, tol=1e-6):
    M = A.conj().T @ A
    lam = max_eig_power(M, n_iter, tol)
    return np.sqrt(np.abs(lam))


def reshape_to_square(vec):
    vec = np.asarray(vec).reshape(-1, 1)
    N = vec.shape[0]
    L = int(np.ceil(np.sqrt(N)))
    pad = L*L - N

    if pad > 0:
        vec_padded = np.vstack([vec, np.zeros((pad, 1))])
    else:
        vec_padded = vec

    return vec_padded.reshape(L, L)


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


def peak_downsample(image, centers, radius):
    """
    Calcula a média dos pixels dentro de círculos centrados em 'centers'.

    Parameters
    ----------
    image : 2D numpy array
        Imagem grayscale
    centers : array-like of shape (N, 2)
        Coordenadas dos centros (y, x)
    radius : int
        Raio do círculo em pixels

    Returns
    -------
    means : numpy array
        Média dos pixels dentro de cada círculo
    """

    means = []

    # máscara circular base
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2

    H, W = image.shape

    for cy, cx in centers:
        cy, cx = int(cy), int(cx)

        # bounding box
        y_min = max(cy - radius, 0)
        y_max = min(cy + radius + 1, H)
        x_min = max(cx - radius, 0)
        x_max = min(cx + radius + 1, W)

        patch = image[y_min:y_max, x_min:x_max]

        # ajuste da máscara (caso esteja na borda)
        mask_crop = mask[
            (y_min - (cy - radius)):(y_max - (cy - radius)),
            (x_min - (cx - radius)):(x_max - (cx - radius))
        ]

        values = patch[mask_crop]

        if values.size > 0:
            means.append(values.mean())
        else:
            means.append(np.nan)

    return np.array(means)


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