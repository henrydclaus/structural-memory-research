"""
Paper C Simulation (Revised, Publication-Oriented)
Structural Memory, Free Energy, and Information in Recurrent Linear Systems

This script generates:
  - fig_energy_decay.png     : homogeneous recurrence (simulation vs closed-form)
  - fig_energy_scaling.png   : forced bounded input (raw + smoothed envelope)
and prints variance verification for white-noise excitation.

Author: Henry Claus
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# 1) GLOBAL PARAMETERS (REVISED)
# ==========================================================

SEED = 7
rng = np.random.default_rng(SEED)

N = 20
alphas = [0.6, 0.9]

# Increase length for better variance agreement, especially for alpha=0.9
T = 30000
burn_in = 5000

# Structural energy truncation depth
# For alpha=0.9, 0.9^300 ~ 2e-14 (negligible), so K=300 is conservative.
K = 300

# Noise std for white-noise test
sigma = 1.0

# Smoothing window for forced-case envelope visualization
SMOOTH_WINDOW = 5 * N  # as used previously


# ==========================================================
# 2) CORE RECURRENCE SIMULATOR
# ==========================================================

def simulate_recurrence(alpha, N, T, u=None, impulse=False):
    """
    Simulates: y[n] = u[n] + alpha*y[n-N], with y[n]=u[n] for n<N.
    If impulse=True: sets y[0]=1 and u[n]=0 (homogeneous recurrence test).
    """
    y = np.zeros(T)

    if impulse:
        u = np.zeros(T)
        y[0] = 1.0
    else:
        if u is None:
            u = np.zeros(T)

    for n in range(T):
        if n >= N:
            y[n] = u[n] + alpha * y[n - N]
        else:
            y[n] = u[n]

    return y


# ==========================================================
# 3) STRUCTURAL FREE ENERGY FUNCTIONAL
# ==========================================================

def structural_energy(y, alpha, N, K):
    """
    Computes:
      F[n] = sum_{k=0}^{K-1} |alpha|^k * |y[n-kN]|^2
    Truncation at K is justified by geometric decay when |alpha|<1.
    """
    T = len(y)
    F = np.zeros(T)
    a = abs(alpha)

    for n in range(T):
        total = 0.0
        # Only terms with n-kN >= 0 contribute
        max_k = min(K, n // N + 1)
        for k in range(max_k):
            idx = n - k * N
            total += (a ** k) * (y[idx] ** 2)
        F[n] = total

    return F


# ==========================================================
# 4) HOMOGENEOUS TEST (SIMULATION vs CLOSED-FORM)
# ==========================================================

def homogeneous_test():
    """
    Homogeneous recurrence: u[n]=0, y[0]=1, y[n<0]=0.
    y[mN]=alpha^m and y[n]=0 otherwise.
    Closed form at n=mN:
      F[mN] = |alpha|^m * (1 - |alpha|^{m+1})/(1 - |alpha|)
    """
    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        y = simulate_recurrence(alpha, N, T, impulse=True)
        F = structural_energy(y, alpha, N, K)

        # Sample at multiples of N for clean comparison
        idx = np.arange(0, T, N)
        m = np.arange(len(idx))

        F_sim = F[idx]

        a = abs(alpha)
        F_th = (a ** m) * (1.0 - a ** (m + 1)) / (1.0 - a)

        # Plot simulation markers and theory dashed line
        plt.plot(idx, F_sim, marker='o', linestyle='None', markersize=4,
                 label=f"Sim α={alpha}")
        plt.plot(idx, F_th, linestyle='--', linewidth=2,
                 label=f"Theory α={alpha}")

    plt.title("Structural Free Energy (Homogeneous)")
    plt.xlabel("n (samples)")
    plt.ylabel("F[n]")

    # Log scale makes both α curves readable across long horizons
    plt.yscale("log")

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_energy_decay.png", dpi=300)
    plt.close()


# ==========================================================
# 5) FORCED BOUNDED INPUT (RAW + SMOOTHED ENVELOPE)
# ==========================================================

def forced_bounded_test():
    """
    Forced recurrence with bounded excitation u[n] ~ Unif[-1,1].
    Plots raw F[n] (transparent) and rolling-mean envelope (solid).
    """
    plt.figure(figsize=(10, 6))

    W = SMOOTH_WINDOW
    kernel = np.ones(W) / W

    for alpha in alphas:
        u = rng.uniform(-1.0, 1.0, T)
        y = simulate_recurrence(alpha, N, T, u=u, impulse=False)
        F = structural_energy(y, alpha, N, K)

        F_smooth = np.convolve(F, kernel, mode='same')

        plt.plot(F, alpha=0.18, linewidth=1.0, label=f"Raw α={alpha}")
        plt.plot(F_smooth, linewidth=2.5, label=f"Smoothed α={alpha}")

    plt.title("Structural Free Energy (Forced Bounded Input)")
    plt.xlabel("n (samples)")
    plt.ylabel("F[n]")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_energy_scaling.png", dpi=300)
    plt.close()


# ==========================================================
# 6) WHITE-NOISE VARIANCE VERIFICATION
# ==========================================================

def variance_test():
    """
    White noise u[n] ~ N(0, sigma^2).
    Theoretical stationary variance:
      Var(y) = sigma^2 / (1 - alpha^2)
    Uses long run + burn-in to reduce bias for alpha near 1.
    """
    print("\nVariance Verification (white-noise input)\n")

    for alpha in alphas:
        u = rng.normal(0.0, sigma, T)
        y = simulate_recurrence(alpha, N, T, u=u, impulse=False)

        y_ss = y[burn_in:]  # discard transient
        empirical_var = np.var(y_ss)
        theoretical_var = sigma**2 / (1.0 - alpha**2)

        print(f"alpha = {alpha}")
        print(f"Empirical variance:   {empirical_var:.6f}")
        print(f"Theoretical variance: {theoretical_var:.6f}")
        print("-" * 44)


# ==========================================================
# 7) MAIN
# ==========================================================

if __name__ == "__main__":
    homogeneous_test()
    forced_bounded_test()
    variance_test()
    print("\nDone. Saved: fig_energy_decay.png, fig_energy_scaling.png\n")
