import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Locked experiment design
# -----------------------------
N = 2**16               # signal length
seed = 7                # fixed seed for reproducibility
np.random.seed(seed)

# Single-delay parameters
alpha = 0.6
Nd = 40

# Multi-delay parameters (sum |alpha_k| < 1 for stability)
alphas = np.array([0.30, 0.25])
Ns = np.array([30, 47])

# -----------------------------
# System simulation (time domain)
# -----------------------------
u = np.random.randn(N)

# Memoryless
y0 = u.copy()

# Single delay: y[n] = u[n] + alpha * y[n-Nd]
y1 = np.zeros(N)
for n in range(N):
    y1[n] = u[n] + (alpha * y1[n - Nd] if n - Nd >= 0 else 0.0)

# Multi-delay: y[n] = u[n] + sum_k alpha_k * y[n-Nk]
yM = np.zeros(N)
for n in range(N):
    acc = 0.0
    for a, Nk in zip(alphas, Ns):
        acc += a * (yM[n - Nk] if n - Nk >= 0 else 0.0)
    yM[n] = u[n] + acc

# -----------------------------
# Frequency response (theoretical, discrete-time)
# H(z) = 1 / (1 - alpha z^{-N})
# -----------------------------
nfft = 8192
w = np.linspace(0, np.pi, nfft, endpoint=True)
z_inv = np.exp(-1j * w)

H0 = np.ones_like(w)  # memoryless: y=u
H1 = 1.0 / (1.0 - alpha * (z_inv ** Nd))
HM = 1.0 / (1.0 - np.sum(alphas[:, None] * (z_inv[None, :] ** Ns[:, None]), axis=0))

# -----------------------------
# Welch-style coherence estimator (NumPy only)
# -----------------------------
def hann(L):
    n = np.arange(L)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (L - 1))

def welch_csd(x, y, fs=1.0, nperseg=2048, noverlap=1024, nfft=4096):
    if noverlap >= nperseg:
        raise ValueError("noverlap must be < nperseg")
    step = nperseg - noverlap
    win = hann(nperseg)
    U = np.sum(win**2)

    nseg = 1 + (len(x) - nperseg) // step
    if nseg <= 0:
        raise ValueError("Signal too short for chosen segment length")

    Pxy = np.zeros(nfft//2 + 1, dtype=np.complex128)
    Pxx = np.zeros(nfft//2 + 1, dtype=np.float64)
    Pyy = np.zeros(nfft//2 + 1, dtype=np.float64)

    for k in range(nseg):
        i0 = k * step
        xs = x[i0:i0+nperseg] * win
        ys = y[i0:i0+nperseg] * win

        X = np.fft.rfft(xs, n=nfft)
        Y = np.fft.rfft(ys, n=nfft)

        Pxy += X * np.conj(Y)
        Pxx += np.abs(X)**2
        Pyy += np.abs(Y)**2

    Pxy /= (nseg * U)
    Pxx /= (nseg * U)
    Pyy /= (nseg * U)

    f = np.fft.rfftfreq(nfft, d=1.0/fs)
    return f, Pxx, Pyy, Pxy

def mscohere(x, y, fs=1.0, nperseg=2048, noverlap=1024, nfft=4096):
    f, Pxx, Pyy, Pxy = welch_csd(x, y, fs, nperseg, noverlap, nfft)
    coh = (np.abs(Pxy)**2) / (Pxx * Pyy + 1e-18)
    return f, coh

# Coherence between input u and each output
fs = 1.0
f0, coh0 = mscohere(u, y0, fs=fs)
f1, coh1 = mscohere(u, y1, fs=fs)
fM, cohM = mscohere(u, yM, fs=fs)

# -----------------------------
# Figure 1: Magnitude response
# -----------------------------
plt.figure()
plt.plot(w, np.abs(H0), label="Memoryless")
plt.plot(w, np.abs(H1), label=f"Single delay (α={alpha}, N={Nd})")
plt.plot(w, np.abs(HM), label=f"Multi delay (α={alphas.tolist()}, N={Ns.tolist()})")
plt.xlabel("ω (rad/sample)")
plt.ylabel("|H(e^{iω})|")
plt.title("Magnitude Response Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("fig1_magnitude_response.png", dpi=200)

# -----------------------------
# Figure 2: Impulse responses (first 300 samples)
# Use u = delta to generate impulse responses directly
# -----------------------------
Limp = 300
u_imp = np.zeros(Limp)
u_imp[0] = 1.0

# single delay impulse
h1 = np.zeros(Limp)
for n in range(Limp):
    h1[n] = u_imp[n] + (alpha * h1[n - Nd] if n - Nd >= 0 else 0.0)

# multi delay impulse
hM = np.zeros(Limp)
for n in range(Limp):
    acc = 0.0
    for a, Nk in zip(alphas, Ns):
        acc += a * (hM[n - Nk] if n - Nk >= 0 else 0.0)
    hM[n] = u_imp[n] + acc

plt.figure()
plt.stem(np.arange(Limp), h1, linefmt='C0-', markerfmt='C0o', basefmt='k-', label="Single delay")
plt.stem(np.arange(Limp), hM, linefmt='C1-', markerfmt='C1s', basefmt='k-', label="Multi delay")
plt.xlabel("n (samples)")
plt.ylabel("h[n]")
plt.title("Impulse Responses (First 300 Samples)")
plt.legend()
plt.tight_layout()
plt.savefig("fig2_impulse_response.png", dpi=200)

# -----------------------------
# Figure 3: Coherence
# -----------------------------
# Plot coherence vs normalized frequency (rad/sample)
omega = 2 * np.pi * f0 / fs  # since fs=1, omega=2πf
plt.figure()
plt.plot(omega, coh0, label="Memoryless")
plt.plot(omega, coh1, label="Single delay")
plt.plot(omega, cohM, label="Multi delay")
plt.xlim(0, np.pi)
plt.ylim(0, 1.05)
plt.xlabel("ω (rad/sample)")
plt.ylabel("γ²_{uy}(ω)")
plt.title("Magnitude-Squared Coherence (Welch Estimate)")
plt.legend()
plt.tight_layout()
plt.savefig("fig3_coherence.png", dpi=200)

print("Saved: fig1_magnitude_response.png, fig2_impulse_response.png, fig3_coherence.png")
print(f"Reproducibility: N={N}, seed={seed}, nfft={nfft}, Welch(nperseg=2048,noverlap=1024,nfft=4096)")
