import jax.numpy as jnp
from jax import vmap
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ


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
    """
    Bilinear interp of p_grid[z_idx,k_idx] to (z_pts, k_pts).
    z_pts, k_pts: shape (n_eval,)
    """
    # find bracketing indices
    iz = jnp.clip(jnp.searchsorted(z_grid, z_pts) - 1, 0, len(z_grid) - 2)
    ik = jnp.clip(jnp.searchsorted(k_grid, k_pts) - 1, 0, len(k_grid) - 2)

    z0 = z_grid[iz];   z1 = z_grid[iz + 1]
    k0 = k_grid[ik];   k1 = k_grid[ik + 1]

    tz = (z_pts - z0) / (z1 - z0)
    tk = (k_pts - k0) / (k1 - k0)

    p00 = p_grid[iz,     ik    ]
    p10 = p_grid[iz + 1, ik    ]
    p01 = p_grid[iz,     ik + 1]
    p11 = p_grid[iz + 1, ik + 1]

    # bilinear
    return (
        (1-tz)*(1-tk)*p00 +
         tz*(1-tk)*p10 +
        (1-tz)* tk *p01 +
         tz* tk *p11
    )



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


def compute_nonlinear_pk(z_vals, cosmo_params, c_min, eta_0):
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

    Returns
    -------
    k : jnp.ndarray
        Wavenumbers in h/Mpc (shape: [n_k]).

    pk : jnp.ndarray
        Nonlinear matter power spectrum (shape: [n_z, n_k]),
        where pk[i, j] = P(k[j], z_vals[i]).
    """
    # Instantiate emulator
    emulator = CPJ(probe='mpk_nonlin')

    # Expand parameters for each z
    omega_b, omega_cdm, h, n_s, ln10As = cosmo_params
    param_array = jnp.stack([
        jnp.full_like(z_vals, omega_b),
        jnp.full_like(z_vals, omega_cdm),
        jnp.full_like(z_vals, h),
        jnp.full_like(z_vals, n_s),
        jnp.full_like(z_vals, ln10As),
        jnp.full_like(z_vals, c_min),
        jnp.full_like(z_vals, eta_0),
        z_vals
    ], axis=1)  # shape (n_z, 8)

    # Predict power spectra
    pk = emulator.predict(param_array)  # shape (n_z, n_k)
    k = emulator.modes  # shape (n_k,)

    return k, pk



def compute_3x2pt_cls(
    nz_lens, nz_source, z, k, p_k, ell,
    delta_z_lens, delta_z_source, m_bias, galaxy_bias,
    h, omega_b, omega_cdm
):
    """
    Compute 3x2pt angular Cl’s under Limber.

    Returns
    -------
    cl_gg : (n_lens,n_lens,n_ell)
    cl_gk : (n_lens,n_source,n_ell)
    cl_kk : (n_source,n_source,n_ell)
    """
    c = 299792.458                   # km/s
    H0 = 100.0 * h                   # km/s/Mpc
    Omega_m = (omega_b + omega_cdm) / h**2
    Omega_L = 1.0 - Omega_m

    # ---- comoving distance chi(z) ----
    Ez_inv = 1.0 / jnp.sqrt(Omega_m*(1+z)**3 + Omega_L)
    dz = jnp.diff(z)
    mid = 0.5*(Ez_inv[:-1] + Ez_inv[1:])
    chi = c/H0 * jnp.concatenate([jnp.array([0.0]), jnp.cumsum(mid * dz)])

    # ---- integration weight for Limber (dz integral) ----
    # integrand factor: (c/H0)*Ez_inv/(chi^2)
    weight_z = (c/H0) * Ez_inv / (chi**2)

    # ---- shift & normalize n(z) ----
    def shift_norm(nz, dzs):
        interp = linear_interp1d(z, nz)
        nz_sh = interp(z + dzs)
        return nz_sh / jax_trapz(nz_sh, z)

    nz_l_sh = vmap(shift_norm)(nz_lens, delta_z_lens)    # (n_lens,n_z)
    nz_s_sh = vmap(shift_norm)(nz_source, delta_z_source)# (n_source,n_z)

    # ---- lensing kernel W_kappa^i(z) ----
    prefac2 = 1.5 * Omega_m * (H0**2)/(c**2) * (1+z) * chi  # (n_z,)

    def make_q(nz_sh_i):
        # for each z_j integrate from j→end: ∫ dz' nz_sh_i(z')*(χ(z')-χ(z_j))/χ(z')
        def Q_j(j):
            chi_j = chi[j]
            chi_slice = chi[j:]
            nz_slice  = nz_sh_i[j:]
            integrand = nz_slice * (chi_slice - chi_j) / chi_slice
            return jax_trapz(integrand, z[j:])
        Q = jnp.array([Q_j(j) for j in range(z.shape[0])])
        return prefac2 * Q

    q_s = vmap(make_q)(nz_s_sh)  # (n_source,n_z)

    # ---- Cl integrand generator ----
def compute_3x2pt_cls(
    nz_lens, nz_source, z, k, p_k, ell,
    delta_z_lens, delta_z_source, m_bias, galaxy_bias,
    h, omega_b, omega_cdm
):
    """
    Compute 3x2pt angular Cl’s under Limber, returning:
      cl_gg: (n_lens,n_lens,n_ell)
      cl_gk: (n_lens,n_source,n_ell)
      cl_kk: (n_source,n_source,n_ell)
    """
    c   = 299792.458            # km/s
    H0  = 100.0 * h             # km/s/Mpc
    Om  = (omega_b + omega_cdm) / h**2
    Ol  = 1.0 - Om

    # ---- compute comoving distance chi(z) ----
    Ez_inv = 1.0 / jnp.sqrt(Om*(1+z)**3 + Ol)      # (n_z,)
    dzs    = jnp.diff(z)
    mid    = 0.5 * (Ez_inv[:-1] + Ez_inv[1:])
    chi    = c/H0 * jnp.concatenate([jnp.array([0.0]), jnp.cumsum(mid * dzs)])  # (n_z,)
    
    # Cut z=0 (chi=0) to avoid 1/chi issues
    z     = z[1:]
    chi   = chi[1:]
    Ez_inv = Ez_inv[1:]
    nz_lens = nz_lens[:, 1:]
    nz_source = nz_source[:, 1:]
    p_k   = p_k[1:, :]

    # ---- Limber weight: (c/H0)*Ez_inv/chi^2, but zero out at chi=0 ----
    w_z = (c/H0) * Ez_inv / (chi**2)
    w_z = jnp.where(chi > 0, w_z, 0.0)             # avoid divide-by-zero

    # ---- shift & normalize n(z) ----
    def shift_norm(nz, dzs):
        f = linear_interp1d(z, nz)
        nzs = f(z + dzs)
        return nzs / jax_trapz(nzs, z)

    nzl_sh = vmap(shift_norm)(nz_lens,   delta_z_lens)   # (n_lens,n_z)
    nzs_sh = vmap(shift_norm)(nz_source, delta_z_source) # (n_source,n_z)

    # ---- build lensing kernel q_i(z) ----
    prefac = 1.5 * Om * (H0**2)/(c**2) * (1+z) * chi       # (n_z,)
    def make_q(nzi):
        # Q_j = ∫_{z_j}^{z_max} dz' nzi(z')*(chi'-chi_j)/chi'
        def Q_j(j):
            chi_j    = chi[j]
            chi_sl   = chi[j:]
            nz_sl    = nzi[j:]
            integrand= nz_sl * (chi_sl - chi_j) / chi_sl
            return jax_trapz(integrand, z[j:])
        Q = jnp.array([Q_j(j) for j in range(z.shape[0])])
        return prefac * Q

    q_s = vmap(make_q)(nzs_sh)  # (n_source,n_z)

    # ---- single-ℓ integrand ----
    '''
    def cl_int(W1, W2, ell_val):
        k_pts  = ell_val / chi                       # (n_z,)
        P_ell  = interp2d_linear(z, k, p_k, z, k_pts) # (n_z,)
        return jax_trapz(W1 * W2 * P_ell * w_z, z)
    '''

    def cl_int(W1, W2, ell_val):
        k_pts  = jnp.where(chi > 0, ell_val / chi, 0.0)
        k_pts  = jnp.clip(k_pts, k[0] + 1e-6, k[-1] - 1e-6)
        P_ell  = interp2d_linear(z, k, p_k, z, k_pts)
        integrand = W1 * W2 * P_ell * w_z
        #print("ell =", ell_val, "integrand =", integrand[:5])
        #print (jax_trapz(integrand, z))
        return jax_trapz(integrand, z)

    # ---- build C_ell arrays ----
    n_l, n_z   = nzl_sh.shape
    n_s, _     = nzs_sh.shape
    n_ell      = ell.shape[0]

    cl_gg = jnp.stack([
        jnp.stack([
            galaxy_bias[i] * galaxy_bias[j] *
            jnp.array([cl_int(nzl_sh[i], nzl_sh[j], L) for L in ell])
        for j in range(n_l)], axis=0)
    for i in range(n_l)], axis=0)

    cl_gk = jnp.stack([
        jnp.stack([
            galaxy_bias[i] * (1 + m_bias[j]) *
            jnp.array([cl_int(nzl_sh[i], q_s[j], L) for L in ell])
        for j in range(n_s)], axis=0)
    for i in range(n_l)], axis=0)

    cl_kk = jnp.stack([
        jnp.stack([
            (1 + m_bias[i]) * (1 + m_bias[j]) *
            jnp.array([cl_int(q_s[i], q_s[j], L) for L in ell])
        for j in range(n_s)], axis=0)
    for i in range(n_s)], axis=0)

    return cl_gg, cl_gk, cl_kk


def flatten_cls_to_vector_jax_safe(cl_gg, cl_gk, cl_kk):
    """
    JAX-safe version: Flatten cl_gg (auto only), cl_gk, and cl_kk into a 1D data vector.

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

    # Extract cl_gg auto terms only (i == j)
    cl_gg_auto = jnp.diagonal(cl_gg, axis1=0, axis2=1)  # (n_ell, n_lens)
    cl_gg_auto = cl_gg_auto.T  # shape (n_lens, n_ell)

    # Reshape everything to (N_block × N_ell,) and concatenate
    gg_flat = cl_gg_auto.reshape(-1)
    gk_flat = cl_gk.reshape(-1)
    kk_flat = cl_kk.reshape(-1)

    data_vector = jnp.concatenate([gg_flat, gk_flat, kk_flat])
    return data_vector


def compute_gaussian_covariance_binned(
    nz_lens, nz_source, z, k, p_k, ell,
    delta_z_lens, delta_z_source, m_bias, galaxy_bias,
    h, omega_b, omega_cdm,
    n_eff, sigma_eps_sq, fsky
):
    """
    Gaussian covariance for binned Cls using Wick's theorem with auto-computed delta_ell.

    Parameters
    ----------
    ell : array (n_ell,)
        Central multipoles of the bins.

    Returns
    -------
    cov : (n_data, n_data) jnp.ndarray
    """
    # --- Auto-compute delta_ell from ell spacing ---
    edges = 0.5 * (ell[1:] + ell[:-1])
    delta_ell = jnp.empty_like(ell)
    delta_ell = delta_ell.at[1:-1].set(edges[1:] - edges[:-1])
    delta_ell = delta_ell.at[0].set(edges[0] - ell[0])
    delta_ell = delta_ell.at[-1].set(ell[-1] - edges[-1])

    # --- Compute Cls ---
    cl_gg, cl_gk, cl_kk = compute_3x2pt_cls(
        nz_lens, nz_source, z, k, p_k, ell,
        delta_z_lens, delta_z_source, m_bias, galaxy_bias,
        h, omega_b, omega_cdm
    )

    n_lens = nz_lens.shape[0]
    n_src  = nz_source.shape[0]
    n_ell  = ell.shape[0]

    # --- Add noise ---
    cl_gg = cl_gg.at[jnp.arange(n_lens), jnp.arange(n_lens)].add(1. / n_eff)
    cl_kk = cl_kk.at[jnp.arange(n_src), jnp.arange(n_src)].add(sigma_eps_sq / n_eff)

    # --- Flatten Cls into data vector (exclude cross-bin gg) ---
    cl_vector = []
    cl_index  = []

    for i in range(n_lens):
        cl_vector.append(cl_gg[i, i])  # shape (n_ell,)
        cl_index.extend([("gg", i, i, l) for l in range(n_ell)])

    for i in range(n_lens):
        for j in range(n_src):
            cl_vector.append(cl_gk[i, j])
            cl_index.extend([("gk", i, j, l) for l in range(n_ell)])

    for i in range(n_src):
        for j in range(n_src):
            cl_vector.append(cl_kk[i, j])
            cl_index.extend([("kk", i, j, l) for l in range(n_ell)])

    cl_vector = jnp.concatenate(cl_vector)
    n_data = len(cl_index)

    # --- Initialize covariance matrix ---
    cov = jnp.zeros((n_data, n_data))

    # --- Wick's theorem ---
    for a in range(n_data):
        t1, i1, j1, l1 = cl_index[a]
        for b in range(n_data):
            t2, i2, j2, l2 = cl_index[b]
            if l1 != l2:
                continue

            ell_val = ell[l1]
            d_ell = delta_ell[l1]
            prefactor = 2. / ((2 * ell_val + 1) * d_ell * fsky)

            def get_cl(type_, i, j, l):
                if type_ == "gg": return cl_gg[i, j, l]
                if type_ == "gk": return cl_gk[i, j, l]
                if type_ == "kk": return cl_kk[i, j, l]

            c1 = get_cl(t1, i1, i2, l1)
            c2 = get_cl(t2, j1, j2, l1)
            c3 = get_cl(t1, i1, j2, l1)
            c4 = get_cl(t2, j1, i2, l1)

            cov = cov.at[a, b].set(prefactor * (c1 * c2 + c3 * c4))

    return cov




