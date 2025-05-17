import jax.numpy as jnp
from jax import vmap
from jax import lax
from jax.scipy.linalg import cho_solve, cho_factor


def linear_interp1d(x, y):
    def interp_fn(x_new):
        idx = jnp.clip(jnp.searchsorted(x, x_new, side='right') - 1, 0, len(x) - 2)
        x0 = x[idx]
        x1 = x[idx + 1]
        y0 = y[idx]
        y1 = y[idx + 1]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x_new - x0)
    return jnp.vectorize(interp_fn)

def jax_trapz(y, x, axis=-1):
    dx = jnp.diff(x)
    mid = 0.5 * (y[..., 1:] + y[..., :-1])
    return jnp.sum(mid * dx, axis=axis)



def interp2d_linear(z_grid, k_grid, p_grid, z_pts, k_pts):
    def interp_single(zi, ki):
        iz = jnp.clip(jnp.searchsorted(z_grid, zi) - 1, 0, len(z_grid) - 2)
        ik = jnp.clip(jnp.searchsorted(k_grid, ki) - 1, 0, len(k_grid) - 2)

        z0 = lax.dynamic_index_in_dim(z_grid, iz, keepdims=False)
        z1 = lax.dynamic_index_in_dim(z_grid, iz + 1, keepdims=False)
        k0 = lax.dynamic_index_in_dim(k_grid, ik, keepdims=False)
        k1 = lax.dynamic_index_in_dim(k_grid, ik + 1, keepdims=False)

        tz = (zi - z0) / (z1 - z0)
        tk = (ki - k0) / (k1 - k0)

        # Extract scalar values
        def get_p(i, j):
            row = lax.dynamic_index_in_dim(p_grid, i, axis=0, keepdims=False)
            return lax.dynamic_index_in_dim(row, j, axis=0, keepdims=False)

        p00 = get_p(iz,     ik)
        p10 = get_p(iz + 1, ik)
        p01 = get_p(iz,     ik + 1)
        p11 = get_p(iz + 1, ik + 1)

        return (1 - tz)*(1 - tk)*p00 + tz*(1 - tk)*p10 + (1 - tz)*tk*p01 + tz*tk*p11

    return vmap(interp_single)(z_pts, k_pts)  # returns shape (n_z,)



def compute_nz_bins(z, n_bins, z0=0.64):
    """
    Compute tomographic n(z) bins with equal galaxy counts.

    Parameters
    ----------
    z : jnp.ndarray
        Redshift grid.
    n_bins : int
        Number of tomographic bins.
    z0 : float
        Characteristic redshift for parent distribution.

    Returns
    -------
    nz : jnp.ndarray
        Array of shape (n_bins, len(z)), each row a normalized tomographic n(z).
    """
    parent_nz = z**2 * jnp.exp(- (z / z0)**1.5)
    parent_nz /= jax_trapz(parent_nz, z)

    # Compute CDF
    dz = jnp.diff(z)
    mid_vals = 0.5 * (parent_nz[:-1] + parent_nz[1:])
    cum_nz = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(mid_vals * dz)])
    cum_nz /= cum_nz[-1]

    # Invert CDF
    inv_cdf = linear_interp1d(cum_nz, z)
    quantiles = jnp.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = inv_cdf(quantiles)

    # Build tomographic bins
    z_broadcast = z[None, :]
    lower_edges = bin_edges[:-1][:, None]
    upper_edges = bin_edges[1:][:, None]
    masks = (z_broadcast >= lower_edges) & (z_broadcast <= upper_edges)

    nz_masked = parent_nz * masks
    norms = jax_trapz(nz_masked, z)
    nz = nz_masked / norms[:, None]

    return nz


def compute_nonlinear_pk(z_vals, cosmo_params, c_min, eta_0, emulator):
    """
    Compute nonlinear matter power spectrum using CosmoPower-JAX.

    Parameters
    ----------
    z_vals : jnp.ndarray
        Array of redshift values (shape: [n_z]).

    cosmo_params : jnp.ndarray
        Cosmological parameters (shape: [5]) in order:
        [omega_b, omega_cdm, h, n_s, ln10^10_A_s].

    c_min : float
        HMCode baryonic feedback parameter c_min.

    eta_0 : float
        HMCode baryonic feedback parameter eta_0.

    emulator : CosmoPowerJAX instance
        Pre-initialized emulator with probe='mpk_nonlin'.

    Returns
    -------
    k : jnp.ndarray
        Wavenumbers in h/Mpc (shape: [n_k]).

    pk : jnp.ndarray
        Nonlinear matter power spectrum (shape: [n_z, n_k]),
        where pk[i, j] = P(k[j], z_vals[i]).
    """
    omega_b, omega_cdm, h, n_s, ln10As = cosmo_params

    # Broadcast params for each redshift
    n_z = z_vals.shape[0]
    param_array = jnp.stack([
        jnp.full((n_z,), omega_b),
        jnp.full((n_z,), omega_cdm),
        jnp.full((n_z,), h),
        jnp.full((n_z,), n_s),
        jnp.full((n_z,), ln10As),
        jnp.full((n_z,), c_min),
        jnp.full((n_z,), eta_0),
        z_vals
    ], axis=1)  # shape (n_z, 8)

    pk = emulator.predict(param_array)  # shape (n_z, n_k)
    k = emulator.modes  # shape (n_k,)

    return k, pk



def compute_3x2pt_cls(
    nz_lens, nz_source, z, k, p_k, ell,
    delta_z_lens, delta_z_source, m_bias, galaxy_bias,
    h, omega_b, omega_cdm
):
    c = 299792.458
    H0 = 100.0 * h
    Omega_m = (omega_b + omega_cdm) / h**2
    Omega_L = 1.0 - Omega_m

    Ez_inv = 1.0 / jnp.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    dz = jnp.diff(z)
    chi = c / H0 * jnp.concatenate([jnp.array([0.0]), jnp.cumsum(0.5 * (Ez_inv[:-1] + Ez_inv[1:]) * dz)])

    # Cut z=0 where chi=0
    z = z[1:]
    chi = chi[1:]
    Ez_inv = Ez_inv[1:]
    nz_lens = nz_lens[:, 1:]
    nz_source = nz_source[:, 1:]
    p_k = p_k[1:, :]

    w_z = (c / H0) * Ez_inv / (chi**2)

    def shift_nz(nz, dz):
        z_shifted = z + dz
        iz = jnp.clip(jnp.searchsorted(z, z_shifted) - 1, 0, len(z) - 2)
        z0 = z[iz]
        z1 = z[iz + 1]
        y0 = nz[iz]
        y1 = nz[iz + 1]
        val = y0 + (y1 - y0) * (z_shifted - z0) / (z1 - z0)
        return val / jax_trapz(val, z)

    nzl_sh = vmap(shift_nz)(nz_lens, delta_z_lens)
    nzs_sh = vmap(shift_nz)(nz_source, delta_z_source)

    prefac = 1.5 * Omega_m * (H0 / c)**2 * (1 + z) * chi

    def make_q(nz_i):
        def Q_j(j):
            # Create slice starting at j
            length = len(z)
            idx = jnp.arange(length)
            mask = idx >= j

            chi_j = chi[j]
            chi_s = chi * mask
            nz_s = nz_i * mask

            # Avoid dividing by zero
            chi_ratio = jnp.where(mask, (chi_s - chi_j) / chi_s, 0.0)

            integrand = nz_s * chi_ratio
            return jax_trapz(integrand, z)

        return prefac * vmap(Q_j)(jnp.arange(len(z)))

    q_s = vmap(make_q)(nzs_sh)

    def cl_integrand(W1, W2, ell_val):
        k_vals = jnp.where(chi > 0, ell_val / chi * h, 0.0) # fixed to make sure chi and k have the same units
        k_vals = jnp.clip(k_vals, k[0] + 1e-6, k[-1] - 1e-6)
        P_ell = interp2d_linear(z, k, p_k, z, k_vals)
        #print("W1", W1.shape, "W2", W2.shape, "P_ell", P_ell.shape, "w_z", w_z.shape)
        return jax_trapz(W1 * W2 * P_ell * w_z, z)

    def cl_gg_fn(i, j):
        return vmap(lambda l: galaxy_bias[i] * galaxy_bias[j] * cl_integrand(nzl_sh[i], nzl_sh[j], l))(ell)

    def cl_gk_fn(i, j):
        return vmap(lambda l: galaxy_bias[i] * (1 + m_bias[j]) * cl_integrand(nzl_sh[i], q_s[j], l))(ell)

    def cl_kk_fn(i, j):
        return vmap(lambda l: (1 + m_bias[i]) * (1 + m_bias[j]) * cl_integrand(q_s[i], q_s[j], l))(ell)

    n_l = nzl_sh.shape[0]
    n_s = nzs_sh.shape[0]

    cl_gg = vmap(lambda i: vmap(lambda j: cl_gg_fn(i, j))(jnp.arange(n_l)))(jnp.arange(n_l))
    cl_gk = vmap(lambda i: vmap(lambda j: cl_gk_fn(i, j))(jnp.arange(n_s)))(jnp.arange(n_l))
    cl_kk = vmap(lambda i: vmap(lambda j: cl_kk_fn(i, j))(jnp.arange(n_s)))(jnp.arange(n_s))

    return cl_gg, cl_gk, cl_kk


def flatten_cls_to_vector_jax_safe(cl_gg, cl_gk, cl_kk):
    """
    JAX-safe: Flatten cl_gg (auto only), cl_gk (only j >= i), and cl_kk (only i <= j)
    into a 1D data vector.

    Parameters
    ----------
    cl_gg : (n_lens, n_lens, n_ell)
    cl_gk : (n_lens, n_source, n_ell)
    cl_kk : (n_source, n_source, n_ell)

    Returns
    -------
    data_vector : jnp.ndarray, shape (n_data,)
    """
    n_lens, _, n_ell = cl_gg.shape
    n_src = cl_kk.shape[0]

    # Auto-only cl_gg
    cl_gg_auto = jnp.diagonal(cl_gg, axis1=0, axis2=1)  # (n_ell, n_lens)
    cl_gg_auto = cl_gg_auto.T  # shape (n_lens, n_ell)
    gg_flat = cl_gg_auto.reshape(-1)

    # cl_gk: only j >= i
    gk_flat_list = []
    for i in range(n_lens):
        for j in range(i, n_src):
            gk_flat_list.append(cl_gk[i, j])
    gk_flat = jnp.concatenate(gk_flat_list, axis=0)

    # cl_kk: only i <= j
    kk_flat_list = []
    for i in range(n_src):
        for j in range(i, n_src):
            kk_flat_list.append(cl_kk[i, j])
    kk_flat = jnp.concatenate(kk_flat_list, axis=0)

    return jnp.concatenate([gg_flat, gk_flat, kk_flat])



def compute_gaussian_covariance_matrix(
    cl_gg, cl_gk, cl_kk, ell, n_eff, sigma_eps_sq, fsky
):
    """
    Compute Gaussian covariance matrix for 3x2pt data vector.
    - Includes cl_gg[i,i]
    - Includes cl_gk[i,j] only when j >= i
    - Includes cl_kk[i,j] only when i <= j
    """

    # Convert to NumPy *after* copying, so we modify noise safely
    import numpy as np
    cl_gg = np.array(cl_gg.copy())
    cl_gk = np.array(cl_gk.copy())
    cl_kk = np.array(cl_kk.copy())
    ell = np.array(ell)

    n_lens, n_src, n_ell = cl_gk.shape[0], cl_kk.shape[0], ell.shape[0]

    # delta_ell bin widths
    edges = 0.5 * (ell[1:] + ell[:-1])
    delta_ell = np.empty_like(ell)
    delta_ell[1:-1] = edges[1:] - edges[:-1]
    delta_ell[0] = edges[0] - ell[0]
    delta_ell[-1] = ell[-1] - edges[-1]

    # Add noise *safely*
    for i in range(n_lens):
        cl_gg[i, i, :] += 1.0 / n_eff
    for i in range(n_src):
        cl_kk[i, i, :] += sigma_eps_sq / n_eff

    # Enforce cl_kk symmetry explicitly
    for i in range(n_src):
        for j in range(i+1, n_src):
            sym = 0.5 * (cl_kk[i, j, :] + cl_kk[j, i, :])
            cl_kk[i, j, :] = sym
            cl_kk[j, i, :] = sym

    # --- Build data vector labels ---
    labels = []

    for i in range(n_lens):
        for l in range(n_ell):
            labels.append(("gg", i, i, l))

    for i in range(n_lens):
        for j in range(i, n_src):
            for l in range(n_ell):
                labels.append(("gk", i, j, l))

    for i in range(n_src):
        for j in range(i, n_src):
            for l in range(n_ell):
                labels.append(("kk", i, j, l))

    n_data = len(labels)
    cov = np.zeros((n_data, n_data))

    def get_cl(tag, i, j, l):
        if tag == "gg":
            return cl_gg[i, j, l]
        elif tag == "gk":
            return cl_gk[i, j, l]
        elif tag == "kk":
            return cl_kk[i, j, l]
        else:
            raise ValueError(f"Unknown tag {tag}")

    for a in range(n_data):
        t1, i1, j1, l1 = labels[a]
        for b in range(a, n_data):
            t2, i2, j2, l2 = labels[b]
            if l1 != l2:
                continue
            ℓ = ell[l1]
            Δℓ = delta_ell[l1]
            prefac = 2.0 / ((2 * ℓ + 1) * Δℓ * fsky)

            c1 = get_cl(t1, i1, i2, l1)
            c2 = get_cl(t2, j1, j2, l1)
            c3 = get_cl(t1, i1, j2, l1)
            c4 = get_cl(t2, j1, i2, l1)

            cov_ab = prefac * (c1 * c2 + c3 * c4)
            cov[a, b] = cov_ab
            cov[b, a] = cov_ab

    return jnp.array(cov)


def gaussian_loglike(
    nz_lens, nz_source, z, k, p_k, ell,
    delta_z_lens, delta_z_source, m_bias, galaxy_bias,
    h, omega_b, omega_cdm,
    cov_inv, data_vector_fid
):
    """
    JAX-safe Gaussian log-likelihood for 3x2pt data vector using explicit inverse.

    Parameters
    ----------
    All inputs match compute_3x2pt_cls, plus:

    cov : (n_data, n_data) ndarray (NumPy or JAX array)
        Precomputed data covariance matrix (assumed constant).

    data_vector_fid : (n_data,) jnp.ndarray
        Fiducial data vector (used as the observed data).

    Returns
    -------
    loglike : float
        Gaussian log-likelihood value.
    """
    # Compute model Cls
    cl_gg, cl_gk, cl_kk = compute_3x2pt_cls(
        nz_lens, nz_source, z, k, p_k, ell,
        delta_z_lens, delta_z_source, m_bias, galaxy_bias,
        h, omega_b, omega_cdm
    )

    # Build model vector
    model_vector = flatten_cls_to_vector_jax_safe(cl_gg, cl_gk, cl_kk)
    delta = model_vector - data_vector_fid

    # Use full inverse instead of Cholesky
    cov_inv = jnp.asarray(cov_inv)
    chi2 = delta @ cov_inv @ delta

    return -0.5 * chi2


def loglike_jax_wrapper(
    theta,              # 1D jnp.ndarray
    n_bins,             # shared for lens/source
    z,                  # redshift array (n_z,)
    ell,                # multipole centers
    cov,                # (n_data, n_data) covariance matrix
    data_vector_fid     # (n_data,) fiducial data vector
):
    """
    JAX-safe log-likelihood wrapper.

    Parameters
    ----------
    theta : jnp.ndarray
        Flat parameter vector:
        [omega_b, omega_cdm, h, n_s, ln10^10_A_s,
         c_min, eta_0,
         galaxy_bias (n_bins),
         delta_z (n_bins),
         m_bias (n_bins)]

    n_bins : int
        Number of tomographic bins (same for lens/source)

    z : jnp.ndarray
        Redshift grid for n(z) and P(k,z)

    ell : jnp.ndarray
        Multipole centers

    cov : jnp.ndarray or np.ndarray
        Covariance matrix

    data_vector_fid : jnp.ndarray
        Fiducial 3x2pt data vector

    Returns
    -------
    loglike : float
        Gaussian log-likelihood value
    """
    idx = 0
    omega_b   = theta[idx]; idx += 1
    omega_cdm = theta[idx]; idx += 1
    h         = theta[idx]; idx += 1
    n_s       = theta[idx]; idx += 1
    ln10As    = theta[idx]; idx += 1
    c_min     = theta[idx]; idx += 1
    eta_0     = theta[idx]; idx += 1

    galaxy_bias = theta[idx : idx + n_bins]; idx += n_bins
    delta_z     = theta[idx : idx + n_bins]; idx += n_bins
    m_bias      = theta[idx : idx + n_bins]; idx += n_bins

    cosmo_params = jnp.array([omega_b, omega_cdm, h, n_s, ln10As])

    # Compute nonlinear P(k, z)
    emulator = CPJ(probe='mpk_nonlin')
    k, p_k = compute_nonlinear_pk(z, cosmo_params, c_min, eta_0, emulator)

    # Recompute n(z)
    nz_lens = compute_nz_bins(z, n_bins, z0=0.64)
    nz_source = compute_nz_bins(z, n_bins, z0=0.64)

    # Compute log-likelihood
    return gaussian_loglike(
        nz_lens, nz_source, z, k, p_k, ell,
        delta_z, delta_z, m_bias, galaxy_bias,
        h, omega_b, omega_cdm,
        cov, data_vector_fid
    )
