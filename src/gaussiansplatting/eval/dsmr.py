import numpy as np
from numba import jit


@jit(nopython=True)
def valnan(u, i, j):
    """Get a pixel (row:j, col:i, channel:c) or a nan."""
    sz = u.shape
    out = np.nan
    if i >= 0 and j >= 0 and i < sz[-1] and j < sz[-2]:
        out = u[j, i]
    return out


@jit(nopython=True)
def downsample2x_(u, out):
    """Downsampling 2x of the image u (numba backend) out must be pre-allocated with half shape."""
    sz = u.shape

    for j in range(sz[0]):
        for i in range(sz[1]):
            v = 0
            count = 0
            vout = np.nan
            for k in range(2):
                for l in range(2):
                    if i + k < sz[1] and j + l < sz[0]:
                        # t = valnan(u, i + k, j + l)
                        t = u[j + l, i + k]
                        if np.isfinite(t):
                            v = v + t
                            count = count + 1
            if count > 0:
                vout = v / count
            out[j // 2, i // 2] = vout
    return out


def downsample2x(u):
    """Downsampling 2x of the image u."""
    sz = u.shape
    out = np.zeros([int(np.ceil(sz[0] / 2)), int(np.ceil(sz[1] / 2))])
    return downsample2x_(u, out)


@jit(
    nopython=True, fastmath=True
)  #! MUCH MUCH FASTER BUT FASTMATH ADD MORE UNCERTAINTY 0.01
def mean_std_fastmath(u, v, dx=0, dy=0):
    """Computes normalized cross correlation coefficient between image u and v shifted by (dx,dy)
    all nan and infinite pixels in the images are ignored.

    Exact same implementation as mean_std_base but with fastmath enabled.This can induce obnoxious
    MAE errors
    """
    sz = u.shape
    muu = 0
    muv = 0
    sigu = 0
    sigv = 0
    xcorr = 0
    count = 0

    # mean
    for j in range(sz[0]):
        for i in range(sz[1]):
            vu = u[j, i]
            if i + dx >= 0 and i + dx < sz[1] and j + dy >= 0 and j + dy < sz[0]:
                vv = v[j + dy, i + dx]
                if np.isfinite(vu) and np.isfinite(vv):
                    muu = muu + vu
                    muv = muv + vv
                    count = count + 1
    muu = muu / count
    muv = muv / count

    # var
    for j in range(sz[0]):
        for i in range(sz[1]):
            vu = u[j, i] - muu
            if i + dx >= 0 and i + dx < sz[1] and j + dy >= 0 and j + dy < sz[0]:
                vv = valnan(v, i + dx, j + dy) - muv
                if np.isfinite(vu) and np.isfinite(vv):
                    sigu = sigu + vu * vu
                    sigv = sigv + vv * vv
                    xcorr = xcorr + vu * vv
    sigu = np.sqrt(sigu / count)
    sigv = np.sqrt(sigv / count)
    xcorr = xcorr / count

    return muu, muv, sigu, sigv, xcorr


@jit(nopython=True)
def mean_std_base(u, v, dx=0, dy=0):
    """Computes normalized cross correlation coefficient between image u and v shifted by (dx,dy)
    all nan and infinite pixels in the images are ignored."""
    sz = u.shape
    muu = 0
    muv = 0
    sigu = 0
    sigv = 0
    xcorr = 0
    count = 0

    # mean
    for j in range(sz[0]):
        for i in range(sz[1]):
            vu = u[j, i]
            if i + dx >= 0 and i + dx < sz[1] and j + dy >= 0 and j + dy < sz[0]:
                vv = v[j + dy, i + dx]
                if np.isfinite(vu) and np.isfinite(vv):
                    muu = muu + vu
                    muv = muv + vv
                    count = count + 1
    muu = muu / count
    muv = muv / count

    # var
    for j in range(sz[0]):
        for i in range(sz[1]):
            vu = u[j, i] - muu
            if i + dx >= 0 and i + dx < sz[1] and j + dy >= 0 and j + dy < sz[0]:
                vv = valnan(v, i + dx, j + dy) - muv
                if np.isfinite(vu) and np.isfinite(vv):
                    sigu = sigu + vu * vu
                    sigv = sigv + vv * vv
                    xcorr = xcorr + vu * vv
    sigu = np.sqrt(sigu / count)
    sigv = np.sqrt(sigv / count)
    xcorr = xcorr / count

    return muu, muv, sigu, sigv, xcorr


@jit(nopython=True)
def ncc(u, v, dx=0, dy=0):
    """Computes normalized cross correlation coefficient between image u and v shifted by (dx,dy)
    all nan and infinite pixels in the images are ignored."""

    _, _, sigu, sigv, xcorr = mean_std_base(u, v, dx, dy)

    return xcorr / (sigu * sigv + 1e-8)


@jit(nopython=True)
def compute_ncc(u, v, irange, initdx, initdy):
    """
    compute the displacement (dx,dy) that maximizes the normalized
    cross correlation between u and v (shifted)
    explores displacements in the range:  (initdx,initdy) +- irange
    """
    best_dx = initdx
    best_dy = initdy
    maxv = -np.inf
    for y in range(initdy - irange, initdy + irange + 1):
        for x in range(initdx - irange, initdx + irange + 1):
            corr = ncc(u, v, x, y)
            if corr > maxv:
                best_dx, best_dy = x, y
                maxv = corr
    return best_dx, best_dy


def recursive_ncc(u, v, irange=5, dx=0, dy=0):
    """Multiscale normalized cross correlation computation."""
    sz = u.shape
    if min(sz[0], sz[1]) > 100:
        su = downsample2x(u)
        sv = downsample2x(v)
        dx = dx // 2
        dy = dy // 2

        dx, dy = recursive_ncc(su, sv, irange, dx, dy)
        dx = dx * 2
        dy = dy * 2

    dx, dy = compute_ncc(u, v, irange, dx, dy)
    return dx, dy


@jit(nopython=True)
def apply_shift_(v, out, dx, dy, a, b, c, d):
    """Apply shift to the numpy array v."""

    sz = v.shape

    for j in range(sz[0]):
        for i in range(sz[1]):
            out[j, i] = a * valnan(v, i + dx, j + dy) + b + c * i + d * j

    return out


### interfaces


def compute_shift(dsm_ref, dsm_sec, scaling=True):
    """Compute the shift needed to register `dsm_sec` on `dsm_ref`

    Args:
        dsm_ref (Array[float]): reference DSM
        dsm_sec (Array[float]): DSM to be registered
        scaling (bool): if True, allow a scaling along the z axis.
            Else, the `a` coefficient will be fixed to 1.

    Returns:
        dx, dy, a, b: shift coefficients to register `dsm_sec` on `dsm_ref`
            `dx` and `dy` are the horizontal shift coefficients
            `a` and `b` are the coefficients of the affine mapping
            `z -> a*z + b` to be applied to the values of `dsm_sec`
    """
    dx, dy = recursive_ncc(dsm_ref, dsm_sec)

    muu, muv, sigu, sigv, xcorr = mean_std_base(dsm_ref, dsm_sec, dx, dy)
    # muu, muv, sigu, sigv, xcorr = mean_std_fastmath(dsm_ref, dsm_sec, dx, dy) # could be used but it failed on IARPA_003 sometimes

    a = sigu / sigv if scaling else 1
    b = muu - muv * a
    # if np.isnan(b):
    #     print("Warning: NaN in computed shift coefficients, fall back to basic jit implementation")
    #     muu,muv, sigu, sigv, xcorr = mean_std_base(dsm_ref, dsm_sec, dx, dy)
    #     a = sigu / sigv if scaling else 1
    #     b = muu - muv * a
    return dx, dy, a, b


def computeshift_benchmarked(dsm_ref, dsm_sec, scaling=True):
    """Compute the shift needed to register `dsm_sec` on `dsm_ref`

    Args:
        dsm_ref (Array[float]): reference DSM
        dsm_sec (Array[float]): DSM to be registered
        scaling (bool): if True, allow a scaling along the z axis.
            Else, the `a` coefficient will be fixed to 1.

    Returns:
        dx, dy, a, b: shift coefficients to register `dsm_sec` on `dsm_ref`
            `dx` and `dy` are the horizontal shift coefficients
            `a` and `b` are the coefficients of the affine mapping
            `z -> a*z + b` to be applied to the values of `dsm_sec`
    """
    import time

    start = time.time()
    dx, dy = recursive_ncc(dsm_ref, dsm_sec)
    print("time to compute ncc: ", time.time() - start)
    start = time.time()

    muu, muv, sigu, sigv, xcorr = mean_std_fastmath(dsm_ref, dsm_sec, dx, dy)

    a = sigu / sigv if scaling else 1
    b = muu - muv * a
    print("time to compute scaling: ", time.time() - start)
    return dx, dy, a, b


def apply_shift(in_dsm, dx=0, dy=0, a=1, b=0, c=0, d=0):
    """Apply a shift of given coefficients to `in_dsm`.

    Args:
        in_dsm (Array[float]): DSM to be shifted
        dx, dy, a, b: shift coefficients to register `dsm_sec` on `dsm_ref`
            `dx` and `dy` are the horizontal shift coefficients
            `a` and `b` are the coefficients of the affine mapping
            `z -> a*z + b` to be applied to the values of `dsm_sec`
    """

    out = np.zeros_like(in_dsm)

    return apply_shift_(in_dsm, out, dx, dy, a, b, c, d)
