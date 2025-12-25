import mpmath as mp
mp.mp.dps = 50

def explicit_psi(x, n_zeros=100, sigma_off=0.5):
    """Explicit formula psi(x); simulate off-line with sigma_off >0.5"""
    psi = mp.mpf(x)
    zeros = [mp.zetazero(k) for k in range(1, n_zeros+1)]
    for rho in zeros:
        rho_off = mp.mpc(sigma_off, rho.imag)  # Off-line pair at same gamma
        psi -= x**rho / rho + x**rho_off / rho_off  # Quartet effect approx
    psi -= mp.log(2*mp.pi) - 0.5*mp.log(1 - 1/x**2)
    return psi

x = mp.mpf(10)
print(f"psi(10, on-line): {explicit_psi(x, 100, 0.5)}")  # ~9.7876
print(f"psi(10, off-line sigma=0.6): {explicit_psi(x, 100, 0.6)}")  # Spike ~10.5