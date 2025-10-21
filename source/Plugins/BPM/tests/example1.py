"""
Basic example of launching an off-center Gaussian beam into a multimode
step index fiber core and watching as the modes beat with each other.

This is the Python equivalent of BPM-Matlab's Example1.m
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..\\..\\..\\..\\Binaries\\RelWithDebInfo\\")
import BPM_py as BPM


def calc_initial_E(X, Y):
    """Define the initial E field - off-center Gaussian beam"""
    w_0 = 2.5e-6  # Beam waist
    offset = 2.5e-6  # Off-center offset
    amplitude = np.exp(-((X - offset) ** 2 + Y**2) / w_0**2)
    return amplitude.astype(np.complex64)


def calc_RI(X, Y, n_background):
    """Define refractive index - step index fiber core"""
    n = np.full(X.shape, n_background, dtype=np.complex64)
    # Core region: circle with 5um radius
    core_mask = X**2 + Y**2 < (5e-6) ** 2
    n[core_mask] = 1.46
    return n


def main():
    # Initialize parameters
    params = BPM.GridParameters()

    # General settings
    params.useGPU = False
    params.useAllCPUs = False

    # Visualization
    params.updates = 100

    # Resolution parameters
    Lx_main = 20e-6
    Ly_main = 20e-6
    padfactor = 1.5
    params.Lx = Lx_main * padfactor
    params.Ly = Ly_main * padfactor
    params.Nx = int(200 * padfactor)
    params.Ny = int(200 * padfactor)

    # These will be auto-calculated in the constructor
    # but you can also set them explicitly:
    # params.dx = params.Lx / params.Nx
    # params.dy = params.Ly / params.Ny
    params.dz = 1e-6  # z step size

    params.alpha = 3e14  # Absorption coefficient

    # Problem definition
    params.lambda_ = 1000e-9  # Wavelength
    params.n_background = 1.45  # Cladding RI
    params.n_0 = 1.46  # Reference RI
    params.Lz = 2e-3  # Propagation distance

    # Symmetry settings (no symmetry in this case)
    params.xSymmetry = 0
    params.ySymmetry = 0

    print("Initializing BPM solver...")
    solver = BPM.BPMSolver(params)

    # Create coordinate grids
    # Note: dx/dy are now available after solver construction
    dx = params.Lx / params.Nx
    dy = params.Ly / params.Ny
    x = dx * (np.arange(params.Nx) - (params.Nx - 1) / 2.0)
    y = dy * (np.arange(params.Ny) - (params.Ny - 1) / 2.0)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initialize refractive index
    print("Initializing refractive index...")
    n_field = calc_RI(X, Y, params.n_background)
    ri = BPM.RefractiveIndex()
    ri.n = n_field.flatten().tolist()
    ri.Nx = params.Nx
    ri.Ny = params.Ny
    ri.Nz = 1  # 2D refractive index
    ri.Lx = params.Lx
    ri.Ly = params.Ly
    ri.xSymmetry = params.xSymmetry
    ri.ySymmetry = params.ySymmetry
    solver.setRI(ri)

    # Initialize electric field
    print("Initializing electric field...")
    E_init = calc_initial_E(X, Y)
    # Normalize
    E_init = E_init / np.sqrt(np.sum(np.abs(E_init) ** 2))

    e_field = BPM.ElectricField()
    e_field.field = E_init.flatten().tolist()
    e_field.Lx = params.Lx
    e_field.Ly = params.Ly
    e_field.xSymmetry = params.xSymmetry
    e_field.ySymmetry = params.ySymmetry
    solver.setE(e_field)

    # Run FD-BPM solver
    print("Running FD-BPM propagation...")
    print(f"  Grid: {params.Nx}x{params.Ny}, dz={params.dz*1e6:.2f} μm")
    print(f"  Propagation length: {params.Lz*1e3:.2f} mm")
    result = solver.propagateFDBPM()

    print(f"\nPropagation complete!")
    print(f"Initial power: {result.powers[0]:.4f}")
    print(f"Final power: {result.powers[-1]:.4f}")
    print(f"Power loss: {(1-result.powers[-1])*100:.2f}%")

    # Plot results
    plot_results(x, y, result, params)


def plot_results(x, y, result, params):
    """Visualize the propagation results"""
    # Reshape final field
    E_final = np.array(result.finalField.field).reshape(params.Nx, params.Ny)
    n_final = np.array(result.finalRI.n).reshape(params.Nx, params.Ny)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Plot 1: Refractive index
    ax = axes[0, 0]
    im = ax.pcolormesh(
        x * 1e6, y * 1e6, np.real(n_final).T, shading="auto", cmap="viridis"
    )
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title("Real part of refractive index")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="n")

    # Plot 2: Power vs z
    ax = axes[0, 1]
    ax.plot(np.array(result.z_positions) * 1e3, result.powers, "b-", linewidth=2)
    ax.set_xlabel("Propagation distance [mm]")
    ax.set_ylabel("Relative power remaining")
    ax.set_ylim([0, 1.1])
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Power evolution")

    # Plot 3: Intensity
    ax = axes[1, 0]
    intensity = np.abs(E_final) ** 2
    # Use log scale for better visualization
    intensity_log = np.log10(intensity + 1e-10)
    im = ax.pcolormesh(x * 1e6, y * 1e6, intensity_log.T, shading="auto", cmap="hot")
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title("Final intensity (log₁₀) [W/m²]")
    ax.set_aspect("equal")

    # Draw fiber core boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    core_radius = 5  # μm
    ax.plot(
        core_radius * np.cos(theta),
        core_radius * np.sin(theta),
        "c--",
        linewidth=2,
        label="Core boundary",
    )
    ax.legend()
    plt.colorbar(im, ax=ax, label="log₁₀(I)")

    # Plot 4: Phase
    ax = axes[1, 1]
    phase = np.angle(E_final)

    im = ax.pcolormesh(
        x * 1e6, y * 1e6, phase.T, shading="auto", cmap="hsv", vmin=-np.pi, vmax=np.pi
    )
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title("Final phase [rad]")
    ax.set_aspect("equal")

    # Draw fiber core boundary
    ax.plot(
        core_radius * np.cos(theta),
        core_radius * np.sin(theta),
        "w--",
        linewidth=2,
        alpha=0.7,
    )
    cbar = plt.colorbar(im, ax=ax, label="Phase [rad]")
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

    plt.suptitle(
        f"BPM Simulation: Off-center Gaussian in MMF\n"
        + f"λ={params.lambda_*1e9:.0f} nm, L={params.Lz*1e3:.1f} mm",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("bpm_example1_results.png", dpi=150, bbox_inches="tight")
    print("Results saved to bpm_example1_results.png")
    plt.show()


if __name__ == "__main__":
    main()
