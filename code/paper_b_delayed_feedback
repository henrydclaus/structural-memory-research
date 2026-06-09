import numpy as np
import matplotlib.pyplot as plt

def impulse_response(N: int, alpha: float, L: int) -> np.ndarray:
    """
    Impulse response h[n] for y[n] = u[n] + alpha y[n-N], with u[n]=delta[n].
    For stable |alpha|<1, h[n] = sum_{k>=0} alpha^k delta[n-kN].
    """
    h = np.zeros(L)
    k = 0
    while k * N < L:
        h[k * N] = alpha ** k
        k += 1
    return h

def magnitude_response(N: int, alpha: float, num_w: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """
    Magnitude response |H(e^{jw})| for H(z)=1/(1 - alpha z^{-N}).
    """
    w = np.linspace(0.0, np.pi, num_w)  # 0..pi is standard for real DT systems
    denom = 1.0 - alpha * np.exp(-1j * w * N)
    H = 1.0 / denom
    mag = np.abs(H)
    return w, mag

def main():
    N = 20
    alphas = [0.6, 0.9]

    # -------- Figure 1: impulse response (stem) --------
    L = 400
    n = np.arange(L)

    plt.figure()
    for alpha in alphas:
        h = impulse_response(N=N, alpha=alpha, L=L)
        # Avoid stem kwargs that older matplotlib may not support
        markerline, stemlines, baseline = plt.stem(n, h, label=rf"$\alpha={alpha}$")
        plt.setp(markerline, markersize=3)

    plt.xlabel("n")
    plt.ylabel("h[n]")
    plt.title(rf"Impulse response for $y[n]=u[n]+\alpha\,y[n-{N}]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_impulse_response.png", dpi=200)
    plt.close()

    # -------- Figure 2: magnitude response --------
    plt.figure()
    for alpha in alphas:
        w, mag = magnitude_response(N=N, alpha=alpha, num_w=4096)
        plt.plot(w, mag, label=rf"$\alpha={alpha}$")

    plt.xlabel(r"$\omega$ (rad/sample)")
    plt.ylabel(r"$|H(e^{i\omega})|$")
    plt.title(rf"Magnitude response for $H(e^{{i\omega}})=1/(1-\alpha e^{{-i\omega {N}}})$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_frequency_response.png", dpi=200)
    plt.close()

    print("Saved: fig_impulse_response.png")
    print("Saved: fig_frequency_response.png")

if __name__ == "__main__":
    main()
