"""Ultrafast and Ultraintense Laser Light - Pulse Characterization.

1. Transform-limited and initially chirped Gaussian pulses
   - temporal intensity and temporal FWHM
   - analytic and numerical Fourier transforms
   - spectral FWHM and time-bandwidth product
2. Intensity autocorrelation of the specified two-component pulse
   - pulse and autocorrelation plots
   - interpolated FWHM values and their ratio
   - Type-II SHG autocorrelator schematic
3. GRENOUILLE
   - component-level operating-principle schematic
   - illustrative single-shot delay-frequency trace

Run in a VS Code terminal:
    python -m pip install numpy matplotlib
    python Pulse_Characterization_Assignment.py

Use ``--no-show`` to generate files without opening plot windows.

FOURIER CONVENTION
------------------
    E(omega) = integral E(t) exp(-i*omega*t) dt.

The assignment defines the chirped field as
    E(t) = E0 exp[-(1 + 2 i alpha)t^2/tau0^2].
Consequently, the spectral-width multiplier is sqrt(1+4 alpha^2).  Some
handwritten supporting pages omit the factor 2; this code follows the printed
assignment exactly and also makes the convention easy to change.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


@dataclass(frozen=True)
class Settings:
    """Numerical inputs. Time is measured in femtoseconds."""

    E0: float = 1.0
    tau0_fs: float = 50.0
    chirp_alpha: float = 1.0
    time_min_fs: float = -500.0
    time_max_fs: float = 500.0
    dt_fs: float = 0.05


def gaussian_field(t: np.ndarray, E0: float, tau0: float,
                   alpha: float = 0.0) -> np.ndarray:
    """Complex Gaussian field from the printed assignment.

    The chirp contributes only temporal phase, so |E(t)|^2 does not depend on
    alpha for this particular parameterization.
    """
    return E0 * np.exp(-(1.0 + 2.0j*alpha)*(t/tau0)**2)


def analytic_spectrum(omega: np.ndarray, E0: float, tau0: float,
                      alpha: float = 0.0) -> np.ndarray:
    """Exact Fourier transform of ``gaussian_field``.

    Integral exp(-A t^2-B t) dt = sqrt(pi/A) exp(B^2/(4A)),
    with A=(1+2i alpha)/tau0^2 and B=i omega.
    """
    q = 1.0 + 2.0j*alpha
    return E0*np.sqrt(np.pi)*tau0/np.sqrt(q) * np.exp(-(omega*tau0)**2/(4.0*q))


def analytic_widths(tau0: float, alpha: float = 0.0) -> dict[str, float]:
    """Return intensity FWHMs and TBP for the assignment field.

    I(t)=E0^2 exp(-2t^2/tau0^2)
        Delta_T = sqrt(2 ln2) tau0.

    |E(omega)|^2 is Gaussian with
        Delta_omega = 2 sqrt(2 ln2) sqrt(1+4alpha^2)/tau0.

    Since Delta_f=Delta_omega/(2pi),
        TBP=Delta_T*Delta_f=(2 ln2/pi)sqrt(1+4alpha^2).
    """
    temporal = np.sqrt(2.0*np.log(2.0))*tau0
    multiplier = np.sqrt(1.0+4.0*alpha**2)
    angular = 2.0*np.sqrt(2.0*np.log(2.0))*multiplier/tau0
    return {"temporal_fwhm_fs": float(temporal),
            "angular_fwhm_rad_per_fs": float(angular),
            "frequency_fwhm_per_fs": float(angular/(2*np.pi)),
            "time_bandwidth_product": float(temporal*angular/(2*np.pi))}


def numerical_fourier(t: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-transform approximation using an FFT and correct scaling."""
    dt = float(t[1]-t[0])
    # NumPy uses exp(-2 pi i kn/N), matching the assignment sign convention.
    spectrum = dt*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))
    cycles_per_fs = np.fft.fftshift(np.fft.fftfreq(t.size, d=dt))
    return 2*np.pi*cycles_per_fs, spectrum


def crossing_x(x1: float, y1: float, x2: float, y2: float,
               level: float) -> float:
    """Linear interpolation of a threshold crossing."""
    if y2 == y1:
        return .5*(x1+x2)
    return x1 + (level-y1)*(x2-x1)/(y2-y1)


def fwhm(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """FWHM of the connected peak containing the global maximum.

    This is more reliable than searching for samples that happen to be close
    to half maximum, particularly for the asymmetric double pulse in Q2.
    """
    y = np.asarray(y, dtype=float)
    peak = int(np.argmax(y)); half = .5*y[peak]
    left = peak
    while left > 0 and y[left] >= half:
        left -= 1
    right = peak
    while right < y.size-1 and y[right] >= half:
        right += 1
    if left == 0 or right == y.size-1:
        raise ValueError("Time/frequency window is too narrow for an FWHM measurement.")
    xl = crossing_x(x[left], y[left], x[left+1], y[left+1], half)
    xr = crossing_x(x[right-1], y[right-1], x[right], y[right], half)
    return float(xr-xl), float(xl), float(xr)


def intensity_autocorrelation(intensity: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute A^(2)(tau)=integral I(t)I(t-tau)dt using zero-padded FFTs."""
    n = intensity.size
    nfft = 1 << (2*n-1).bit_length()
    transform = np.fft.fft(intensity, nfft)
    circular = np.fft.ifft(transform*np.conj(transform)).real
    correlation = np.r_[circular[-(n-1):], circular[:n]]*dt
    delay = np.arange(-(n-1), n)*dt
    return delay, correlation


def plot_gaussian_exercise(output: Path, s: Settings) -> dict[str, float]:
    t = np.arange(s.time_min_fs, s.time_max_fs+s.dt_fs/2, s.dt_fs)
    cases = [(0.0, "transform limited"), (s.chirp_alpha, fr"chirped, $\alpha={s.chirp_alpha:g}$")]
    colors = ["#2468b4", "#d44747"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    results: dict[str, float] = {}
    for (alpha, label), color in zip(cases, colors):
        field = gaussian_field(t, s.E0, s.tau0_fs, alpha)
        intensity = abs(field)**2
        omega, numeric = numerical_fourier(t, field)
        analytic = analytic_spectrum(omega, s.E0, s.tau0_fs, alpha)
        numeric_spectral = abs(numeric)**2; numeric_spectral /= numeric_spectral.max()
        analytic_spectral = abs(analytic)**2; analytic_spectral /= analytic_spectral.max()
        visible_frequency = abs(omega) <= .16
        widths = analytic_widths(s.tau0_fs, alpha)
        temporal_num, tl, tr = fwhm(t, intensity)
        spectral_num, wl, wr = fwhm(omega, numeric_spectral)
        key = "TL" if alpha == 0 else "chirped"
        results[f"{key}_temporal_fwhm_fs"] = temporal_num
        results[f"{key}_spectral_fwhm_rad_per_fs"] = spectral_num
        results[f"{key}_TBP"] = temporal_num*spectral_num/(2*np.pi)
        results[f"{key}_analytic_TBP"] = widths["time_bandwidth_product"]

        axes[0, 0].plot(t, intensity/intensity.max(), color=color, lw=2, label=label)
        temporal_phase = -2.0*alpha*(t/s.tau0_fs)**2
        axes[1, 0].plot(t, temporal_phase, color=color, lw=1.8, label=label)
        axes[0, 1].plot(omega[visible_frequency], analytic_spectral[visible_frequency],
                        color=color, lw=2, label=label)
        axes[0, 1].plot(omega[visible_frequency][::2], numeric_spectral[visible_frequency][::2], ".", color=color,
                        ms=2.5, alpha=.45)
        spectral_phase = (
            alpha*(omega*s.tau0_fs)**2/(2.0*(1.0+4.0*alpha**2))
            - .5*np.arctan(2.0*alpha)
        )
        axes[1, 1].plot(omega[visible_frequency], spectral_phase[visible_frequency],
                        color=color, lw=1.8, label=label)

    temporal_theory = analytic_widths(s.tau0_fs, 0)["temporal_fwhm_fs"]
    axes[0, 0].axhline(.5, color="gray", ls=":")
    axes[0, 0].set(xlim=(-2.5*s.tau0_fs, 2.5*s.tau0_fs), xlabel="time (fs)",
                   ylabel="normalized intensity",
                   title=fr"Temporal intensity, $\Delta T_{{FWHM}}={temporal_theory:.3f}$ fs")
    axes[1, 0].set(xlim=(-2.5*s.tau0_fs, 2.5*s.tau0_fs), xlabel="time (fs)",
                   ylabel="phase (rad)", title="Exact quadratic temporal phase")
    axes[0, 1].axhline(.5, color="gray", ls=":")
    axes[0, 1].set(xlim=(-.16, .16), xlabel=r"angular frequency detuning $\omega$ (rad/fs)",
                   ylabel="normalized spectral intensity",
                   title="Analytic curves and numerical-FFT points")
    axes[1, 1].set(xlim=(-.16, .16), xlabel=r"angular frequency detuning $\omega$ (rad/fs)",
                   ylabel="spectral phase (rad)", title="Exact spectral phase")
    for ax in axes.flat:
        ax.grid(alpha=.24); ax.legend(fontsize=9)
    fig.suptitle("Exercise 1: transform-limited and initially chirped Gaussian pulses", fontsize=16)
    fig.savefig(output/"Exercise1_Gaussian_pulses.png", dpi=240)

    alphas = np.linspace(0, 3, 400)
    tbp = np.array([analytic_widths(s.tau0_fs, a)["time_bandwidth_product"] for a in alphas])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(alphas, tbp, lw=2.5)
    axes[0].axhline(2*np.log(2)/np.pi, color="black", ls="--", label="Gaussian minimum")
    axes[0].set(xlabel=r"chirp parameter $\alpha$", ylabel=r"$\Delta T\Delta f$",
                title=r"TBP = $(2\ln2/\pi)\sqrt{1+4\alpha^2}$")
    # Display numerical versus analytic width for a sweep of chirp values.
    test_alpha = np.linspace(0, 2, 21)
    numerical_widths = []
    exact_widths = []
    for alpha in test_alpha:
        omega, spec = numerical_fourier(t, gaussian_field(t, s.E0, s.tau0_fs, alpha))
        numerical_widths.append(fwhm(omega, abs(spec)**2)[0])
        exact_widths.append(analytic_widths(s.tau0_fs, alpha)["angular_fwhm_rad_per_fs"])
    axes[1].plot(test_alpha, exact_widths, lw=2.5, label="analytic")
    axes[1].plot(test_alpha, numerical_widths, "o", ms=4, label="FFT")
    axes[1].set(xlabel=r"chirp parameter $\alpha$", ylabel=r"$\Delta\omega_{FWHM}$ (rad/fs)",
                title="Spectral broadening verification")
    for ax in axes:
        ax.grid(alpha=.24); ax.legend()
    fig.savefig(output/"Exercise1_time_bandwidth_product.png", dpi=240)
    results["maximum_FFT_width_relative_error"] = float(np.max(
        abs(np.asarray(numerical_widths)-exact_widths)/np.asarray(exact_widths)))
    return results


def plot_autocorrelation_exercise(output: Path, s: Settings) -> dict[str, float]:
    t = np.arange(s.time_min_fs, s.time_max_fs+s.dt_fs/2, s.dt_fs)
    # Exact intensity specified in Exercise 2 (t in fs).
    intensity = np.exp(-(t/50.0)**2) + .4*np.exp(-((t-100.0)/50.0)**2)
    delay, ac = intensity_autocorrelation(intensity, s.dt_fs)
    pulse_width, pulse_left, pulse_right = fwhm(t, intensity)
    ac_width, ac_left, ac_right = fwhm(delay, ac)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].plot(t, intensity, color="#2468b4", lw=2.4)
    axes[0].axhline(.5*intensity.max(), color="gray", ls="--")
    axes[0].axvspan(pulse_left, pulse_right, color="#2468b4", alpha=.12,
                    label=fr"FWHM = {pulse_width:.2f} fs")
    axes[0].set(xlim=(-250, 350), xlabel="time (fs)", ylabel="intensity",
                title="Specified asymmetric two-component pulse")
    axes[1].plot(delay, ac/ac.max(), color="#c23b37", lw=2.4)
    axes[1].axhline(.5, color="gray", ls="--")
    axes[1].axvspan(ac_left, ac_right, color="#c23b37", alpha=.12,
                    label=fr"FWHM = {ac_width:.2f} fs")
    axes[1].set(xlim=(-500, 500), xlabel="delay (fs)",
                ylabel="normalized autocorrelation",
                title=fr"$A^{{(2)}}(\tau)=\int I(t)I(t-\tau)dt$; ratio = {ac_width/pulse_width:.3f}")
    for ax in axes:
        ax.grid(alpha=.24); ax.legend(fontsize=10)
    fig.suptitle("Exercise 2: intensity autocorrelation", fontsize=16)
    fig.savefig(output/"Exercise2_pulse_and_autocorrelation.png", dpi=240)
    return {"Q2_pulse_FWHM_fs": pulse_width,
            "Q2_autocorrelation_FWHM_fs": ac_width,
            "Q2_AC_to_pulse_FWHM_ratio": ac_width/pulse_width}


def draw_autocorrelator_setup(output: Path) -> None:
    """Conceptual Type-II SHG scanning autocorrelator used in Exercise 2b."""
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    ax.set(xlim=(0, 14), ylim=(-3, 4)); ax.axis("off")
    blue = "#2468b4"; red = "#c23b37"
    ax.add_patch(Rectangle((.3, -.45), 1.2, .9, fc="#59636f", ec="black"))
    ax.text(.9, -.8, "input pulse", ha="center", fontsize=11, weight="bold")
    ax.plot([1.5, 3], [0, 0], color=blue, lw=3)
    ax.add_patch(Rectangle((2.9, -.75), .18, 1.5, angle=35, fc="#a7d8ea", ec="#1c647d"))
    ax.text(3, -1.25, "beam splitter", ha="center", fontsize=10)
    # Fixed and variable-delay arms.
    ax.plot([3.1, 5.7, 7.6], [.15, 2.2, .25], color=blue, lw=2.8)
    ax.plot([3.1, 5.5, 7.6], [-.15, -2.0, -.25], color=red, lw=2.8)
    for x, y in [(5.7, 2.2), (5.5, -2.0)]:
        ax.add_patch(Rectangle((x-.12, y-.55), .24, 1.1, angle=45, fc="#cbd0d6", ec="black"))
    ax.add_patch(FancyArrowPatch((4.5, -2.55), (6.6, -2.55), arrowstyle="<->",
                                 mutation_scale=14, color="black"))
    ax.text(5.55, -2.9, "variable delay tau", ha="center", fontsize=10)
    ax.add_patch(Polygon([[7.5, -.75], [8.3, 0], [7.5, .75]], closed=True,
                         fc="#d997e5", ec="#6a2875"))
    ax.text(7.9, -1.25, "Type-II SHG\ncrystal", ha="center", fontsize=10, weight="bold")
    ax.plot([8.3, 10.2], [0, 0], color="#7b3294", lw=3)
    ax.add_patch(Rectangle((10.2, -.7), .25, 1.4, fc="#f0c35a", ec="#765c15"))
    ax.text(10.3, -1.25, "SHG filter", ha="center", fontsize=10)
    ax.plot([10.45, 12], [0, 0], color="#7b3294", lw=3)
    ax.add_patch(Rectangle((12, -.65), 1.1, 1.3, fc="#79b36a", ec="#275d21"))
    ax.text(12.55, -1.1, "slow detector", ha="center", fontsize=10, weight="bold")
    ax.text(7, 3.45, "Scan delay -> overlap in crystal -> integrate SHG energy",
            ha="center", fontsize=15, weight="bold", color="#173f5f")
    ax.text(7, 2.9,
            "Key considerations: polarization alignment, temporal/spatial overlap, thin-crystal bandwidth,\n"
            "phase matching, stable delay scan, background subtraction, and autocorrelation ambiguity",
            ha="center", fontsize=11)
    fig.savefig(output/"Exercise2_TypeII_autocorrelator_setup.png", dpi=240)


def draw_grenouille(output: Path) -> None:
    """Explain GRENOUILLE components and show an illustrative FROG-like trace."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    ax = axes[0]; ax.set(xlim=(0, 12), ylim=(-4, 4)); ax.axis("off")
    ax.add_patch(Rectangle((.2, -.5), 1.2, 1, fc="#59636f", ec="black"))
    ax.text(.8, -1, "wide input\npulse", ha="center", fontsize=10, weight="bold")
    ax.plot([1.4, 3], [0, 0], color="#2468b4", lw=4)
    ax.add_patch(Polygon([[3, -1.2], [3.5, 0], [3, 1.2], [2.6, 0]],
                         fc="#bde9f6", ec="#1d6c88"))
    ax.text(3, -1.75, "Fresnel biprism\nsplits + crosses beams", ha="center", fontsize=9)
    ax.plot([3.5, 5.7], [.25, 1.4], color="#2468b4", lw=2.8)
    ax.plot([3.5, 5.7], [-.25, -1.4], color="#d44747", lw=2.8)
    ax.add_patch(Rectangle((5.7, -2.2), .7, 4.4, fc="#d997e5", ec="#6a2875", lw=2))
    ax.text(6.05, -2.8, "thick SHG crystal\ntime gate + spectrometer", ha="center", fontsize=9)
    # Cylindrical lens and camera.
    ax.add_patch(Rectangle((7.6, -2), .25, 4, fc="#a7d8ea", ec="#1c647d"))
    ax.text(7.72, -2.55, "cylindrical\nlens", ha="center", fontsize=9)
    ax.plot([6.4, 9.5], [0, 0], color="#7b3294", lw=3)
    ax.add_patch(Rectangle((9.5, -2.1), 1.3, 4.2, fc="#454b52", ec="black"))
    ax.text(10.15, -2.65, "2-D camera", ha="center", fontsize=10, weight="bold")
    ax.text(6, 3.5, "GRENOUILLE: GRating-Eliminated No-nonsense Observation of\nUltrafast Incident Laser Light E-fields",
            ha="center", fontsize=13, weight="bold", color="#173f5f")

    # Illustrative delay-frequency trace: not a reconstruction algorithm, but
    # it demonstrates the two dimensions recorded by the camera.
    delay = np.linspace(-3, 3, 500); freq = np.linspace(-3, 3, 500)
    D, F = np.meshgrid(delay, freq)
    trace = np.exp(-D**2/1.5) * np.exp(-(F-.38*D**2)**2/.7)
    trace += .28*np.exp(-(D/1.8)**2) * np.exp(-((F+1.3)/.55)**2)
    im = axes[1].imshow(trace, extent=[delay[0], delay[-1], freq[0], freq[-1]],
                        origin="lower", aspect="auto", cmap="magma")
    axes[1].set(xlabel="relative delay mapped horizontally",
                ylabel="frequency mapped vertically",
                title="Illustrative single-shot GRENOUILLE trace")
    fig.colorbar(im, ax=axes[1], label="SHG signal")
    fig.savefig(output/"Exercise3_GRENOUILLE.png", dpi=240)


def export_summary(output: Path, values: dict[str, float]) -> None:
    with (output/"pulse_characterization_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["quantity", "value"])
        for key, value in values.items(): writer.writerow([key, f"{value:.12g}"])


def run(output: Path, s: Settings, show: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    q1 = plot_gaussian_exercise(output, s)
    q2 = plot_autocorrelation_exercise(output, s)
    draw_autocorrelator_setup(output)
    draw_grenouille(output)
    values = {**q1, **q2}
    export_summary(output, values)

    print("\nPULSE CHARACTERIZATION ASSIGNMENT")
    print("="*55)
    print("Exercise 1 - Gaussian pulse")
    print(f"  Temporal FWHM                  = {q1['TL_temporal_fwhm_fs']:.4f} fs")
    print(f"  Transform-limited spectral FWHM = {q1['TL_spectral_fwhm_rad_per_fs']:.6f} rad/fs")
    print(f"  Transform-limited TBP         = {q1['TL_TBP']:.6f}")
    print(f"  Chirped spectral FWHM         = {q1['chirped_spectral_fwhm_rad_per_fs']:.6f} rad/fs")
    print(f"  Chirped TBP (alpha={s.chirp_alpha:g})       = {q1['chirped_TBP']:.6f}")
    print(f"  Maximum FFT width error       = {100*q1['maximum_FFT_width_relative_error']:.4f}%")
    print("Exercise 2 - specified pulse")
    print(f"  Pulse FWHM                     = {q2['Q2_pulse_FWHM_fs']:.4f} fs")
    print(f"  Autocorrelation FWHM           = {q2['Q2_autocorrelation_FWHM_fs']:.4f} fs")
    print(f"  AC/pulse FWHM ratio            = {q2['Q2_AC_to_pulse_FWHM_ratio']:.6f}")
    print("Exercise 3 - GRENOUILLE schematic and trace generated.")
    print(f"\nResults saved in: {output.resolve()}")
    if show: plt.show()
    else: plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau0", type=float, default=50.0, help="Gaussian field tau0 in fs")
    parser.add_argument("--alpha", type=float, default=1.0, help="chirp parameter alpha")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent/"results_pulse_characterization")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.tau0 <= 0: parser.error("--tau0 must be positive")
    run(args.output, Settings(tau0_fs=args.tau0, chirp_alpha=args.alpha),
        show=not args.no_show)


if __name__ == "__main__":
    main()
