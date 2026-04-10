import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# ── Parameters ─────────────────────────────────────────────────────────────────
alpha = 1.1   # prey birth rate
beta  = 0.4   # predation rate
delta = 0.1   # predator growth rate (from eating prey)
gamma = 0.4   # predator death rate

x0, y0 = 10.0, 10.0   # initial populations
t_end  = 200.0
dt     = 0.1

# ── math ───────────────────────────────────────────────────────
def jjmath(state, alpha, beta, delta, gamma):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])

# ── RK4 integrator ────────────────────────────────────────────────────────────
def rk4(state, dt, *params):
    k1 = jjmath(state,           *params)
    k2 = jjmath(state + dt/2*k1, *params)
    k3 = jjmath(state + dt/2*k2, *params)
    k4 = jjmath(state + dt   *k3, *params)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ── Run simulation ────────────────────────────────────────────────────────────
steps  = int(t_end / dt)
t_arr  = np.linspace(0, t_end, steps)
states = np.zeros((steps, 2))
states[0] = [x0, y0]

for i in range(steps - 1):
    states[i+1] = rk4(states[i], dt, alpha, beta, delta, gamma)

x_arr, y_arr = states[:, 0], states[:, 1]

# ── Multiple initial conditions for phase portrait ────────────────────────────
ic_list = [
    (5, 5), (10, 10), (15, 5), (8, 15), (20, 8), (3, 8)
]
phase_curves = []
for xi, yi in ic_list:
    s = np.zeros((steps, 2))
    s[0] = [xi, yi]
    for i in range(steps - 1):
        s[i+1] = rk4(s[i], dt, alpha, beta, delta, gamma)
    phase_curves.append(s)

# ── Conserved quantity V(x,y) ─────────────────────────────────────────────────
def V(x, y):
    return delta*x - gamma*np.log(x) + beta*y - alpha*np.log(y)

# ── Styling ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
PREY    = "#58d68d"
PRED    = "#e74c3c"
GRID    = "#21262d"
TEXT    = "#e6edf3"
ACCENT  = "#ffd166"
PHASE_COLORS = plt.cm.plasma(np.linspace(0.15, 0.85, len(ic_list)))

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "text.color":        TEXT,
    "font.family":       "monospace",
})

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                        left=0.07, right=0.97, top=0.90, bottom=0.08)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle("Predator–Prey  Simulation",
             fontsize=18, fontweight="bold", color=TEXT, y=0.97,
             fontfamily="monospace")
fig.text(0.5, 0.935,
         f"α={alpha}  β={beta}  δ={delta}  γ={gamma}   "
         f"x₀={x0}  y₀={y0}   RK4  Δt={dt}",
         ha="center", fontsize=9, color="#8b949e")


ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(PANEL)
ax1.plot(t_arr, x_arr, color=PREY, lw=1.8, label="Prey (chickens)")
ax1.plot(t_arr, y_arr, color=PRED, lw=1.8, label="Predators (foxes)")
ax1.fill_between(t_arr, x_arr, alpha=0.08, color=PREY)
ax1.fill_between(t_arr, y_arr, alpha=0.08, color=PRED)
ax1.set_xlabel("Time", fontsize=11)
ax1.set_ylabel("Population", fontsize=11)
ax1.set_title("Population Dynamics Over Time", fontsize=13, pad=10, color=TEXT)
ax1.legend(fontsize=10, framealpha=0.15, facecolor=PANEL, edgecolor=GRID)
ax1.grid(True, alpha=0.4)
ax1.set_xlim(0, t_end)

# annotate period
peaks = np.where((x_arr[1:-1] > x_arr[:-2]) & (x_arr[1:-1] > x_arr[2:]))[0] + 1
if len(peaks) >= 2:
    period = t_arr[peaks[1]] - t_arr[peaks[0]]
    ax1.annotate(f"T ≈ {period:.2f}",
                 xy=(t_arr[peaks[0]], x_arr[peaks[0]]),
                 xytext=(t_arr[peaks[0]] + 0.3, x_arr[peaks[0]] + 2),
                 fontsize=8, color=ACCENT,
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(PANEL)
for curve, color, ic in zip(phase_curves, PHASE_COLORS, ic_list):
    ax2.plot(curve[:, 0], curve[:, 1], color=color, lw=1.2, alpha=0.85)
    ax2.plot(*ic, "o", color=color, ms=4)

# equilibrium point
x_eq = gamma / delta
y_eq = alpha / beta
ax2.plot(x_eq, y_eq, "*", color=ACCENT, ms=12, zorder=5, label=f"Eq ({x_eq:.1f}, {y_eq:.1f})")
ax2.set_xlabel("Prey Population", fontsize=10)
ax2.set_ylabel("Predator Population", fontsize=10)
ax2.set_title("Phase Portrait", fontsize=13, pad=10, color=TEXT)
ax2.legend(fontsize=9, framealpha=0.15, facecolor=PANEL, edgecolor=GRID)
ax2.grid(True, alpha=0.4)

plt.show()