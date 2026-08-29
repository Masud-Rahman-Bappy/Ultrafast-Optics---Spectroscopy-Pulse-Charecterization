from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Parameters:
    """Input parameters and units for all three questions."""

    # Q1: using time in fs makes angular frequency have units rad/fs.
    T2_fs: float = 3.0
    omega_a_rad_per_fs: float = 10.0
    E0: float = 1.0
    delays_fs: tuple[float, ...] = (-0.5, -1.0, -2.0, -5.0)

    # Q2: representative EO-crystal parameters.  The assignment leaves
    # epsilon, chi and E_THz symbolic, so convenient values are used for plots.
    epsilon_r: float = 10.0
    # Small-signal range: large enough to show EO modulation without many
    # phase wraps in a 0.5-mm crystal.
    chi_E_max: float = 3.0e-4    # maximum value of the product chi*E_THz
    probe_wavelength_m: float = 800e-9
    eo_thickness_m: float = 0.5e-3

    # Q3: values stated in the assignment.
    xray_beam_width_m: float = 1.0e-3
    time_window_s: float = 1.0e-12
    temporal_resolution_s: float = 10.0e-15
    speed_of_light: float = 299_792_458.0


# =============================================================================
# QUESTION 1 - PERTURBED FREE-INDUCTION DECAY
# =============================================================================

def susceptibility_time(t_fs: np.ndarray, T2_fs: float,
                        omega_a: float) -> np.ndarray:
    """Causal response chi(t)=Theta(t) exp(-t/T2) exp(-i*omega_a*t)."""
    return np.where(t_fs >= 0.0,
                    np.exp(-t_fs / T2_fs) * np.exp(-1j * omega_a * t_fs),
                    0.0j)


def quenched_susceptibility_time(t_fs: np.ndarray, tau_fs: float,
                                 T2_fs: float, omega_a: float) -> np.ndarray:
    """Susceptibility after an instantaneous pump-induced quench.

    The assignment writes chi'(t)=[1-Theta(t+tau)] chi(t).
    Therefore:
      * tau >= 0: the pump arrives before/at the delta-function probe and the
        response is completely suppressed;
      * tau < 0: the response exists only for 0 <= t < |tau|.
    """
    base = susceptibility_time(t_fs, T2_fs, omega_a)
    if tau_fs >= 0:
        return np.zeros_like(base)
    return np.where((t_fs >= 0) & (t_fs < abs(tau_fs)), base, 0.0j)


def susceptibility_frequency(detuning: np.ndarray, T2_fs: float) -> np.ndarray:
    """Analytic Fourier transform using F(omega)=int f(t)e^(+i omega t)dt.

    detuning = omega-omega_a and chi(omega)=1/(1/T2-i*detuning).
    """
    return 1.0 / (1.0 / T2_fs - 1j * detuning)


def quenched_susceptibility_frequency(detuning: np.ndarray, tau_fs: float,
                                      T2_fs: float) -> np.ndarray:
    """Analytic transform of the finite-duration FID."""
    if tau_fs >= 0:
        return np.zeros_like(detuning, dtype=complex)
    duration = abs(tau_fs)
    denominator = 1.0 / T2_fs - 1j * detuning
    return (1.0 - np.exp(-denominator * duration)) / denominator


def detected_intensity(chi_omega: np.ndarray, E0: float) -> np.ndarray:
    """Thin-sample approximation I=|E|^2+2 Re(E* P), with P=chi E."""
    return E0**2 * (1.0 + 2.0 * np.real(chi_omega))


def plot_question1(output: Path, p: Parameters) -> dict[str, float]:
    """Create time-domain and frequency-domain FID figures."""
    t = np.linspace(-1.5, 9.0, 5000)
    base = susceptibility_time(t, p.T2_fs, p.omega_a_rad_per_fs)
    tau_example = p.delays_fs[2]
    truncated = quenched_susceptibility_time(
        t, tau_example, p.T2_fs, p.omega_a_rad_per_fs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(t, base.real, label="Re[chi(t)]", lw=2)
    axes[0, 0].plot(t, base.imag, "--", label="Im[chi(t)]", lw=2)
    axes[0, 0].set_title("Unperturbed causal free-induction decay")
    axes[0, 1].plot(t, truncated.real, label="Re[chi'(t)]", lw=2)
    axes[0, 1].plot(t, truncated.imag, "--", label="Im[chi'(t)]", lw=2)
    axes[0, 1].axvline(abs(tau_example), color="black", ls=":",
                       label=fr"quench at $|\tau|={abs(tau_example):g}$ fs")
    axes[0, 1].set_title("Pump-truncated response, tau < 0")
    positive = quenched_susceptibility_time(t, 1.0, p.T2_fs,
                                             p.omega_a_rad_per_fs)
    axes[1, 0].plot(t, positive.real, color="#7b3294", lw=2)
    axes[1, 0].plot([], [], color="#7b3294", label="chi'(t)=0")
    axes[1, 0].set_title("Pump precedes probe, tau > 0: response quenched")
    axes[1, 1].plot(t, np.abs(base), label="|chi(t)|", lw=2)
    axes[1, 1].plot(t, np.abs(truncated), "--", label="|chi'(t)|", lw=2)
    axes[1, 1].set_title("Response envelopes")
    for ax in axes.flat:
        ax.set(xlabel="time (fs)", ylabel="normalized susceptibility")
        ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.suptitle("Question 1C: time-domain perturbed FID", fontsize=16)
    fig.savefig(output / "Q1_time_domain_FID.png", dpi=240)

    detuning = np.linspace(-12.0, 12.0, 6001)
    chi = susceptibility_frequency(detuning, p.T2_fs)
    intensity_0 = detected_intensity(chi, p.E0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             constrained_layout=True)
    axes[0].plot(detuning, intensity_0, color="black", lw=2.5,
                 label="unpumped I(omega)")
    integrated_changes = []
    for tau in p.delays_fs:
        chi_q = quenched_susceptibility_frequency(detuning, tau, p.T2_fs)
        intensity_q = detected_intensity(chi_q, p.E0)
        delta_i = intensity_q - intensity_0
        axes[0].plot(detuning, intensity_q, lw=1.3,
                     label=fr"tau={tau:g} fs")
        axes[1].plot(detuning, delta_i, lw=1.5,
                     label=fr"tau={tau:g} fs")
        integrated_changes.append(float(np.trapezoid(delta_i, detuning)))
    axes[0].set(ylabel="detected intensity (normalized)",
                title="Unpumped and pump-quenched spectra")
    axes[1].set(xlabel=r"detuning $\omega-\omega_a$ (rad/fs)",
                ylabel=r"$\Delta I(\omega)$",
                title="Oscillatory spectral artifact caused by time truncation")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8)
    fig.savefig(output / "Q1_frequency_domain_spectra.png", dpi=240)

    # A finite spectrometer range does not integrate the oscillation exactly.
    # Increasing the range approaches the ideal all-frequency result.
    tau_scan = np.linspace(-8, 2, 500)
    finite_band_signal = []
    for tau in tau_scan:
        iq = detected_intensity(
            quenched_susceptibility_frequency(detuning, tau, p.T2_fs), p.E0)
        finite_band_signal.append(np.trapezoid(iq-intensity_0, detuning))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(tau_scan, finite_band_signal, lw=2)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set(xlabel="pump delay tau (fs)", ylabel="finite-band integrated Delta I",
           title="Wavelength-integrated response over the simulated bandwidth")
    ax.grid(alpha=.25)
    fig.savefig(output / "Q1_integrated_time_response.png", dpi=240)
    return {"largest_abs_finite_band_integral": max(map(abs, integrated_changes))}


# =============================================================================
# QUESTION 2 - ELECTRO-OPTIC SAMPLING
# =============================================================================

def rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def crystal_tensor(epsilon: float, chi: float,
                   field_crystal: np.ndarray) -> np.ndarray:
    """Dielectric tensor in the crystallographic basis."""
    Ex, Ey, Ez = field_crystal
    return np.array([[epsilon, chi*Ez, chi*Ey],
                     [chi*Ez, epsilon, chi*Ex],
                     [chi*Ey, chi*Ex, epsilon]])


def lab_tensor(epsilon: float, chi_E: float) -> np.ndarray:
    """Expected lab tensor for E_THz parallel to lab x=(xc+yc)/sqrt(2)."""
    return np.array([[epsilon, 0.0, chi_E],
                     [0.0, epsilon, 0.0],
                     [chi_E, 0.0, epsilon]])


def verify_tensor_rotation(epsilon: float, chi_E: float) -> float:
    """Rotate the crystal tensor by 45 degrees and compare with Eq. Q2A."""
    # E_lab=(E,0,0), and the inverse 45-degree rotation gives equal crystal
    # components Ex_c=Ey_c=E/sqrt(2). We set chi=1, E=chi_E.
    theta = np.pi/4
    R = rotation_z(theta)
    # For x_lab=(x_c+y_c)/sqrt(2), both crystal-frame components are positive.
    field_crystal = np.array([chi_E/np.sqrt(2), chi_E/np.sqrt(2), 0.0])
    eps_c = crystal_tensor(epsilon, 1.0, field_crystal)
    # Passive basis change used in the assignment: eps_lab=R(-theta) eps_c R(theta).
    rotated = R.T @ eps_c @ R
    return float(np.max(np.abs(rotated - lab_tensor(epsilon, chi_E))))


def plot_question2(output: Path, p: Parameters) -> dict[str, float]:
    chi_E = np.linspace(-p.chi_E_max, p.chi_E_max, 800)
    n_minus = np.sqrt(p.epsilon_r - chi_E)
    n_plus = np.sqrt(p.epsilon_r + chi_E)
    delta_phi = (2*np.pi/p.probe_wavelength_m) * (n_plus-n_minus) * p.eo_thickness_m

    # A balanced EO detector operated near quadrature has normalized difference
    # S=(I1-I2)/(I1+I2)=sin(delta_phi), linear for small THz fields.
    I1 = .5*(1 + np.sin(delta_phi))
    I2 = .5*(1 - np.sin(delta_phi))
    balanced = I1-I2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    n0 = np.sqrt(p.epsilon_r)
    axes[0].plot(chi_E, n_minus-n0, label=r"$n_1=\sqrt{\epsilon-\chi E_{THz}}$", lw=2)
    axes[0].plot(chi_E, n_plus-n0, label=r"$n_2=\sqrt{\epsilon+\chi E_{THz}}$", lw=2)
    axes[0].set(xlabel=r"electro-optic perturbation $\chi E_{THz}$",
                ylabel=r"index change $n-\sqrt{\epsilon}$",
                title="THz-induced eigenindex splitting")
    axes[1].plot(chi_E, I1, label="analyzer channel 1", lw=1.6)
    axes[1].plot(chi_E, I2, label="analyzer channel 2", lw=1.6)
    axes[1].plot(chi_E, balanced, color="black", label="balanced difference", lw=2.4)
    axes[1].set(xlabel=r"electro-optic perturbation $\chi E_{THz}$",
                ylabel="normalized signal", title="Quarter-wave plate + balanced analyzer")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.suptitle("Question 2: electro-optic sampling response", fontsize=16)
    fig.savefig(output / "Q2_electro_optic_sampling.png", dpi=240)
    # Save the result but do not leave this figure open for plt.show().
    plt.close(fig)

    test = lab_tensor(p.epsilon_r, .25)
    eigvals, eigvecs = np.linalg.eigh(test)
    error = verify_tensor_rotation(p.epsilon_r, .25)
    return {"tensor_rotation_max_error": error,
            "eigenvalue_min": float(eigvals[0]),
            "eigenvalue_center": float(eigvals[1]),
            "eigenvalue_max": float(eigvals[2]),
            "eigenvector_orthogonality_error": float(np.max(abs(eigvecs.T@eigvecs-np.eye(3))))}


# =============================================================================
# QUESTION 3 - FEL ARRIVAL-TIME SPATIAL ENCODING
# =============================================================================

def spatial_encoding_design(p: Parameters) -> dict[str, float]:
    """Calculate a feasible single-shot spatial-encoding geometry.

    For beams crossing at angle alpha, delay changes along coordinate x as
        Delta t(x) = x*sin(alpha)/c.
    Requiring a window T across width w gives sin(alpha)=c*T/w.
    The required spatial sampling is dx=c*dt/sin(alpha).  With the assignment
    values this is 10 micrometres, so a detector with <=10 um object-plane
    sampling meets the 10 fs requirement.  The complementary angle relative
    to the sample surface is 90-alpha, approximately 72.5 degrees, consistent
    with the convention used in the supplied solution.
    """
    ratio = p.speed_of_light*p.time_window_s/p.xray_beam_width_m
    if ratio > 1:
        raise ValueError("Requested time window is impossible for this beam width.")
    alpha = np.arcsin(ratio)
    dx = p.speed_of_light*p.temporal_resolution_s/np.sin(alpha)
    bins = p.xray_beam_width_m/dx
    # If a material response is sampled along propagation, c*dt is the
    # equivalent longitudinal resolution (3 um for 10 fs).
    longitudinal_resolution = p.speed_of_light*p.temporal_resolution_s
    return {"crossing_angle_deg": float(np.degrees(alpha)),
            "complementary_sample_angle_deg": float(90-np.degrees(alpha)),
            "required_object_pixel_m": float(dx),
            "required_time_bins": float(bins),
            "longitudinal_resolution_m": float(longitudinal_resolution)}


def plot_question3(output: Path, p: Parameters,
                   design: dict[str, float]) -> None:
    width_um = p.xray_beam_width_m*1e6
    x_um = np.linspace(0, width_um, 1600)
    sigma_um = max(design["required_object_pixel_m"]*1e6/2.355, 1.0)
    arrival_delays_fs = [-300, 0, 300]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    # A finite-resolution error-function-like step is approximated with tanh.
    for delay_fs in arrival_delays_fs:
        edge_um = (
            width_um/2
            + delay_fs*1e-15*p.speed_of_light
            / np.sin(np.radians(design["crossing_angle_deg"]))*1e6
        )
        normalized_transmission = .5*(1 + np.tanh((x_um-edge_um)/(np.sqrt(2)*sigma_um)))
        axes[0].plot(x_um, normalized_transmission, lw=2,
                     label=f"arrival shift {delay_fs:+d} fs")
    axes[0].set(xlabel="CCD object-plane coordinate (um)",
                ylabel=r"normalized $I_{ON}/I_{OFF}$",
                title="Arrival-time jitter encoded as a moving spatial edge")
    axes[0].grid(alpha=.25); axes[0].legend()

    # Draw a quantitative crossing geometry.
    alpha = np.radians(design["crossing_angle_deg"])
    axes[1].plot([0, 1], [0, 0], color="#2468b4", lw=8, label="X-ray footprint (1 mm)")
    axes[1].arrow(.05, -.35, .9*np.cos(alpha), .9*np.sin(alpha),
                  width=.012, head_width=.06, length_includes_head=True,
                  color="#d44747", label="optical probe pulse front")
    axes[1].text(.5, .08, "X-ray excited region", ha="center", fontsize=11)
    axes[1].text(.49, -.18,
        f"time window = 1 ps\ncrossing angle = {design['crossing_angle_deg']:.2f} deg\n"
        f"required sampling <= {1e6*design['required_object_pixel_m']:.2f} um",
        ha="center", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=.9))
    axes[1].set(xlim=(-.05, 1.08), ylim=(-.45, .65), aspect="equal",
                title="Feasible spatial-encoding geometry")
    axes[1].axis("off"); axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Question 3: single-shot FEL arrival-time monitor", fontsize=16)
    fig.savefig(output / "Q3_FEL_spatial_encoding.png", dpi=240)
    # Save the result but do not leave this figure open for plt.show().
    plt.close(fig)


def export_summary(output: Path, rows: dict[str, float]) -> None:
    with (output / "assignment_numerical_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["quantity", "value"])
        for key, value in rows.items():
            writer.writerow([key, f"{value:.12g}"])


def run(output: Path, p: Parameters, show: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    q1 = plot_question1(output, p)
    q2 = plot_question2(output, p)
    q3 = spatial_encoding_design(p)
    plot_question3(output, p, q3)
    summary = {**q1, **q2, **q3}
    export_summary(output, summary)

    print("\nECE ultrafast-spectroscopy assignment results")
    print("="*52)
    print("Q1: analytic perturbed-FID spectra and time responses generated.")
    print(f"    Finite simulated bandwidth residual = {q1['largest_abs_finite_band_integral']:.3e}")
    print("Q2: rotated dielectric tensor and eigenproblem checked.")
    print(f"    Tensor rotation error = {q2['tensor_rotation_max_error']:.3e}")
    print(f"    Eigenvalues = {q2['eigenvalue_min']:.6g}, {q2['eigenvalue_center']:.6g}, {q2['eigenvalue_max']:.6g}")
    print("Q3: feasible FEL spatial-encoding design:")
    print(f"    beam crossing angle       = {q3['crossing_angle_deg']:.3f} deg")
    print(f"    angle from sample surface = {q3['complementary_sample_angle_deg']:.3f} deg")
    print(f"    object-plane sampling     <= {1e6*q3['required_object_pixel_m']:.3f} um")
    print(f"    longitudinal resolution  <= {1e6*q3['longitudinal_resolution_m']:.3f} um")
    print(f"    independent time bins     = {q3['required_time_bins']:.1f}")
    print(f"\nResults saved in: {output.resolve()}")
    if show:
        plt.show()
    else:
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results_ultrafast_assignment")
    parser.add_argument("--no-show", action="store_true",
                        help="save figures without opening plot windows")
    args = parser.parse_args()
    run(args.output, Parameters(), show=not args.no_show)


if __name__ == "__main__":
    main()
