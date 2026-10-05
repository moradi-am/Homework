"""
Homework #1 - Turbulence Modeling (F. Sabetghadam, Fall 2026)
The Lorenz system integrated with the forward Euler method.

Runs:
  Part 1: (s, r, b) = (10, 15, 8/3),  IC = (0, 1, 0)      -> periodic/steady
  Part 2: (s, r, b) = (10, 50, 8/3),  IC = (0, 1, 0)      -> chaotic
  Part 3: both parameter sets,        IC = (0, 1.001, 0)  -> sensitivity test

t in [0, 1000], dt = 0.01  (100,001 steps per run)
Figures are written to ./figures/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

S, R_PER, R_CHAO, B = 10.0, 15.0, 50.0, 8.0 / 3.0
IC1 = (0.0, 1.0, 0.0)
IC2 = (0.0, 1.001, 0.0)
T_END, DT = 1000.0, 0.01


def rhs(x, y, z, s, r, b):
    """Lorenz vector field f(x)."""
    return s * (y - x), r * x - y - x * z, x * y - b * z


def euler(s, r, b, ic, t_end=T_END, dt=DT):
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    X = np.empty((n + 1, 3))
    X[0] = ic
    for i in range(n):
        x, y, z = X[i]
        dx, dy, dz = rhs(x, y, z, s, r, b)
        X[i + 1, 0] = x + dt * dx
        X[i + 1, 1] = y + dt * dy
        X[i + 1, 2] = z + dt * dz
    return t, X


def fixed_points(r, b):
    """Analytical non-trivial fixed points C+/- of the Lorenz system."""
    v = np.sqrt(b * (r - 1.0))
    return np.array([v, v, r - 1.0]), np.array([-v, -v, r - 1.0])


def fig_timeseries(t, X, title, fname):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, comp, lab, col in zip(axes, "xyz", ["x(t)", "y(t)", "z(t)"],
                                  ["#1f77b4", "#d62728", "#2ca02c"]):
        ax.plot(t, X[:, "xyz".index(comp)], color=col, lw=0.8)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def fig_phases(t, X, title, fname, stride=25, stride3d=8):
    xs, ys, zs = X[::stride, 0], X[::stride, 1], X[::stride, 2]
    ts = t[::stride]
    fig = plt.figure(figsize=(10, 9))
    planes = [("x", "y"), ("x", "z"), ("y", "z")]
    for i, (a, b_) in enumerate(planes):
        ax = fig.add_subplot(2, 2, i + 1)
        u = {"x": xs, "y": ys, "z": zs}[a]
        v = {"x": xs, "y": ys, "z": zs}[b_]
        sc = ax.scatter(u, v, c=ts, cmap="viridis", s=4, alpha=0.7)
        ax.scatter([u[0]], [v[0]], c="red", s=60, marker="*", zorder=5, label="start")
        ax.scatter([u[-1]], [v[-1]], c="black", s=40, marker="x",
                   zorder=5, label="end")
        ax.set_xlabel(a)
        ax.set_ylabel(b_)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.scatter(X[::stride3d, 0], X[::stride3d, 1], X[::stride3d, 2],
               c=t[::stride3d], cmap="viridis", s=2, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def main():
    print("Integrating (dt = %.3f, t in [0, %.0f]) ..." % (DT, T_END))
    t, Xp = euler(S, R_PER, B, IC1)    # Part 1
    _, Xc = euler(S, R_CHAO, B, IC1)   # Part 2
    _, Xp2 = euler(S, R_PER, B, IC2)    # Part 3 (r = 15)
    _, Xc2 = euler(S, R_CHAO, B, IC2)  # Part 3 (r = 50)

    # ---- figures ----
    fig_timeseries(t, Xp, "Part 1: periodic case (s, r, b) = (10, 15, 8/3), IC=(0,1,0)",
                   "fig1_p1_timeseries.png")
    fig_phases(t, Xp, "Part 1: phase portraits and 3D trajectory (r = 15)",
               "fig2_p1_phases.png")
    fig_timeseries(t, Xc, "Part 2: chaotic case (s, r, b) = (10, 50, 8/3), IC=(0,1,0)",
                   "fig3_p2_timeseries.png")
    fig_phases(t, Xc, "Part 2: phase portraits and 3D trajectory (r = 50)",
               "fig4_p2_phases.png")

    # ---- Part 3 comparison figure ----
    sep_p = np.linalg.norm(Xp - Xp2, axis=1)
    sep_c = np.linalg.norm(Xc - Xc2, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    m = t <= 60
    axes[0, 0].plot(t[m], Xp[m, 0], lw=1, label="IC=(0,1,0)")
    axes[0, 0].plot(t[m], Xp2[m, 0], lw=1, label="IC=(0,1.001,0)")
    axes[0, 0].set_title("r = 15: x(t), both ICs (t in [0,60])")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(t[m], Xc[m, 0], lw=1, label="IC=(0,1,0)")
    axes[0, 1].plot(t[m], Xc2[m, 0], lw=1, label="IC=(0,1.001,0)")
    axes[0, 1].set_title("r = 50: x(t), both ICs (t in [0,60])")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].semilogy(t, sep_p, color="#2ca02c", lw=1)
    axes[1, 0].set_title("r = 15: separation ||X1-X2||(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].semilogy(t, sep_c, color="#d62728", lw=1)
    axes[1, 1].set_title("r = 50: separation ||X1-X2||(t)")
    axes[1, 1].set_xlabel("t")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Part 3: sensitivity to initial conditions")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_p3_sensitivity.png"), dpi=150)
    plt.close(fig)

    # ---- verification numbers ----
    cp, cm = fixed_points(R_PER, B)
    num_fp = Xp[-int(0.1 * len(t)):].mean(axis=0)  # mean of last 10%
    nearer = "C+" if np.linalg.norm(num_fp - cp) < np.linalg.norm(num_fp - cm) else "C-"
    print("\n=== Part 1 (r = 15) ===")
    print("Analytical C+ :", np.round(cp, 6), "  C- :", np.round(cm, 6))
    print("Trajectory settled on:", nearer)
    print("Numerical tail mean:", np.round(num_fp, 6))
    print("Final state       :", np.round(Xp[-1], 6))
    tgt = cp if nearer == "C+" else cm
    print("|num - %s|       :" % nearer, np.round(np.abs(num_fp - tgt), 8))

    print("\n=== Part 2 (r = 50) ===")
    for i, lab in enumerate("xyz"):
        print(f"{lab}: min={Xc[:, i].min():9.4f}  max={Xc[:, i].max():9.4f}  "
              f"mean={Xc[:, i].mean():9.4f}")
    print("Any NaN/inf:", not np.all(np.isfinite(Xc)))

    print("\n=== Part 3 separation ===")
    print("r=15: initial=%.3e  at t=100=%.3e  final=%.3e"
          % (sep_p[0], sep_p[int(100 / DT)], sep_p[-1]))
    print("r=50: initial=%.3e  at t=10=%.3e  at t=100=%.3e  final=%.3e"
          % (sep_c[0], sep_c[int(10 / DT)], sep_c[int(100 / DT)], sep_c[-1]))
    # exponential growth rate estimate for r=50 (fit on t in [2, 12])
    sel = (t >= 2) & (t <= 12)
    lam = np.polyfit(t[sel], np.log(sep_c[sel]), 1)[0]
    print("r=50: estimated divergence rate lambda ~= %.3f (1/time)" % lam)
    print("\nFigures written to:", OUT)


if __name__ == "__main__":
    main()
