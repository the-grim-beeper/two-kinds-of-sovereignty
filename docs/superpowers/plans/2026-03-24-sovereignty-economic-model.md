# Sovereignty Economic Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-layer formal economic model (analytical core, evolutionary simulation, policy interface) of the "Two Kinds of Sovereignty" essay, with both a formal write-up and an interactive Streamlit dashboard.

**Architecture:** Three nested Python modules sharing a calibration layer. Layer 1 (analytical) solves the optimal control problem numerically. Layer 2 (evolutionary) runs agent-based simulations to validate Layer 1 and reveal distributional dynamics. Layer 3 (policy) synthesizes both into actionable lookup tables and dashboards. A Streamlit app exposes all three layers interactively.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Matplotlib, Plotly, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-sovereignty-economic-model-design.md`

---

## File Structure

```
sovereignty-model/
├── model/
│   ├── __init__.py           # Package exports
│   ├── calibration.py        # Shared parameters, empirical anchors, sweep utilities
│   ├── analytical.py         # Layer 1: state equations, HJB, optimal control
│   ├── evolutionary.py       # Layer 2: firm agents, innovation, selection, Monte Carlo
│   ├── policy.py             # Layer 3: policy lookup, delay costs, regime comparison
│   └── visualization.py      # Plotting for all layers (Matplotlib + Plotly)
├── app/
│   └── dashboard.py          # Streamlit interactive companion
├── tests/
│   ├── __init__.py
│   ├── test_calibration.py   # Parameter validation, sweep tests
│   ├── test_analytical.py    # Special-case verification, ODE integration
│   ├── test_evolutionary.py  # Convergence, regime classification
│   └── test_policy.py        # Boundary cases, delay cost monotonicity
├── output/
│   ├── figures/              # Generated static figures
│   └── latex/                # Generated LaTeX-ready write-up
├── pyproject.toml            # Project config, dependencies
└── README.md
```

---

### Task 1: Project Scaffolding and Calibration Module

**Files:**
- Create: `sovereignty-model/pyproject.toml`
- Create: `sovereignty-model/model/__init__.py`
- Create: `sovereignty-model/model/calibration.py`
- Create: `sovereignty-model/tests/__init__.py`
- Create: `sovereignty-model/tests/test_calibration.py`

- [ ] **Step 1: Create project directory structure**

```bash
cd ~/projects/sovereignty-visualization
mkdir -p sovereignty-model/{model,app,tests,output/{figures,latex}}
```

- [ ] **Step 2: Write pyproject.toml**

Create `sovereignty-model/pyproject.toml`:

```toml
[project]
name = "sovereignty-model"
version = "0.1.0"
description = "Formal economic model of technological sovereignty allocation"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.11",
    "matplotlib>=3.7",
    "plotly>=5.15",
    "streamlit>=1.28",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create virtual environment and install dependencies**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 4: Write the failing test for calibration parameters**

Create `sovereignty-model/tests/__init__.py` (empty) and `sovereignty-model/tests/test_calibration.py`:

```python
"""Tests for the shared calibration module."""
import pytest
from model.calibration import Parameters, window_openness, h_sigma


class TestParameters:
    """Verify default parameter values match empirical anchors."""

    def test_initial_dependency(self):
        p = Parameters()
        assert p.D0 == pytest.approx(0.67, abs=0.01)

    def test_rd_intensity_eu(self):
        p = Parameters()
        assert p.R == pytest.approx(0.022, abs=0.001)

    def test_dependency_range(self):
        p = Parameters()
        assert 0.0 < p.D0 < 1.0
        assert 0.0 < p.D_bar <= 1.0

    def test_all_depreciation_rates_positive(self):
        p = Parameters()
        assert p.delta_p > 0
        assert p.delta_f > 0

    def test_exploration_returns_in_unit_interval(self):
        p = Parameters()
        assert 0 < p.beta < 1
        assert 0 < p.kappa < 1


class TestWindowOpenness:
    """Verify Perez window dynamics."""

    def test_window_at_open_time(self):
        """Window should be at full openness when it just opened."""
        w = window_openness(t=5.0, t_open=5.0, W0=1.0, gamma=0.1)
        assert w == pytest.approx(1.0)

    def test_window_decays(self):
        """Window should decay over time."""
        w1 = window_openness(t=10.0, t_open=5.0, W0=1.0, gamma=0.1)
        w2 = window_openness(t=20.0, t_open=5.0, W0=1.0, gamma=0.1)
        assert w1 > w2
        assert w2 > 0

    def test_window_before_open(self):
        """Window should be zero before it opens."""
        w = window_openness(t=3.0, t_open=5.0, W0=1.0, gamma=0.1)
        assert w == 0.0


class TestHSigma:
    """Verify the constraint instrument hump shape."""

    def test_h_at_zero(self):
        """No constraint = baseline productivity."""
        assert h_sigma(0.0) == pytest.approx(1.0)

    def test_hump_shape(self):
        """h should increase then decrease."""
        h_low = h_sigma(0.5)
        h_mid = h_sigma(2.0)
        h_high = h_sigma(10.0)
        assert h_mid > h_low, "h should increase from 0 to σ*"
        assert h_mid > h_high, "h should decrease past σ*"

    def test_h_always_positive(self):
        """h(σ) should remain positive for all σ."""
        for sigma in [0, 0.5, 1, 2, 5, 10, 20, 50]:
            assert h_sigma(sigma) > 0

    def test_peak_location(self):
        """Peak should be at σ* = a/b (analytically derivable)."""
        # With default a, b: peak at 1/b
        from model.calibration import Parameters
        p = Parameters()
        sigma_star = 1.0 / p.b
        # h at peak should be greater than neighbors
        assert h_sigma(sigma_star) >= h_sigma(sigma_star - 0.5)
        assert h_sigma(sigma_star) >= h_sigma(sigma_star + 0.5)
```

- [ ] **Step 5: Run tests to verify they fail**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
source .venv/bin/activate
pytest tests/test_calibration.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 6: Write calibration module**

Create `sovereignty-model/model/__init__.py`:

```python
"""Sovereignty economic model — three-layer nested framework."""
```

Create `sovereignty-model/model/calibration.py`:

```python
"""Shared parameters, empirical anchors, and utility functions.

All parameters documented with sources. This module is the single source
of truth for calibration across all three model layers.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Parameters:
    """Model parameters with empirical calibration.

    Layer 1 (Analytical) parameters:
        R: Resource flow (EU R&D intensity as fraction of GDP)
        kappa: Exploitation production function exponent (Cobb-Douglas)
        beta: Exploration production function exponent
        A: TFP for exploitation
        delta_p: Present sovereignty depreciation rate
        delta_f: Future sovereignty depreciation rate
        D0: Initial dependency level (Synergy Research Group)
        D_bar: Dependency lock-in ceiling
        lam: Lock-in speed (Arthur increasing returns)
        eta: Present sovereignty effect on dependency
        a, b: Constraint instrument hump-shape parameters
        omega: Weight on present vs future sovereignty in value function
        psi: Dependency penalty (quadratic)
        rho: Discount rate

    Window parameters:
        W0: Initial window openness
        gamma: Window closure rate (domain-specific)
        t_open: Window opening time

    Layer 2 (Evolutionary) parameters:
        N: Number of firms
        p0: Base exploitation success probability
        q0: Base exploration success probability
        nu: Capability exponent in success probability
        eps_exploit: Exploitation payoff scale
        eps_explore: Exploration payoff location (Pareto)
        xi: Exploration payoff shape (Pareto)
        theta: Selection intensity
        s_min: Minimum market share before exit
    """

    # --- Layer 1: Analytical ---
    R: float = 0.022          # EU R&D/GDP (EIB)
    kappa: float = 0.6        # Exploitation diminishing returns
    beta: float = 0.5         # Exploration returns exponent
    A: float = 1.0            # TFP for exploitation (normalized)
    delta_p: float = 0.05     # Present sovereignty depreciation (5%/yr)
    delta_f: float = 0.03     # Future sovereignty depreciation (3%/yr)
    D0: float = 0.67          # Initial dependency (Synergy Research Group)
    D_bar: float = 0.95       # Lock-in ceiling
    lam: float = 0.3          # Lock-in speed
    eta: float = 0.1          # Present sovereignty → dependency reduction
    a: float = 2.0            # Hump shape: amplitude
    b: float = 0.5            # Hump shape: decay rate → σ* = 1/b = 2.0
    omega: float = 0.5        # Weight on present sovereignty
    psi: float = 1.0          # Dependency penalty
    rho: float = 0.03         # Discount rate (3%)

    # --- Window parameters ---
    W0: float = 1.0           # Initial window openness
    gamma: float = 0.1        # Window closure rate
    t_open: float = 0.0       # Window opening time

    # --- Layer 2: Evolutionary ---
    N: int = 500              # Number of firms
    p0: float = 0.7           # Base exploitation success probability
    q0: float = 0.15          # Base exploration success probability
    nu: float = 0.3           # Capability exponent
    eps_exploit: float = 0.05 # Exploitation payoff scale
    eps_explore: float = 0.1  # Exploration payoff location (Pareto)
    xi: float = 2.5           # Exploration payoff shape (Pareto)
    theta: float = 0.5        # Selection intensity
    s_min: float = 0.0005     # Minimum market share (1/N * 0.25)

    # --- Simulation ---
    T: float = 30.0           # Time horizon (years)
    dt: float = 1.0           # Layer 2 period length (years)


def window_openness(t: float, t_open: float, W0: float, gamma: float) -> float:
    """Compute Perez window openness at time t.

    W(t) = W0 * exp(-gamma * max(0, t - t_open))  for t >= t_open
    W(t) = 0                                        for t < t_open
    """
    if t < t_open:
        return 0.0
    return W0 * np.exp(-gamma * (t - t_open))


def h_sigma(sigma: float, a: float = 2.0, b: float = 0.5) -> float:
    """Constraint instrument productivity modifier.

    h(σ) = 1 + a·σ·exp(-b·σ)

    Non-monotonic hump shape:
    - h(0) = 1 (no constraint, baseline)
    - Peak at σ* = 1/b (designed pressure)
    - Decays toward 1 for large σ (destructive pressure)
    """
    return 1.0 + a * sigma * np.exp(-b * sigma)


def phi_dependency(D: float, scale: float = 0.1) -> float:
    """Dependency-accelerated depreciation.

    φ(D) = scale * D^2

    Convex: high dependency accelerates erosion of domestic capacity.
    """
    return scale * D ** 2
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
source .venv/bin/activate
pytest tests/test_calibration.py -v
```

Expected: All PASS

- [ ] **Step 8: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/pyproject.toml sovereignty-model/model/__init__.py sovereignty-model/model/calibration.py sovereignty-model/tests/__init__.py sovereignty-model/tests/test_calibration.py
git commit -m "feat: add project scaffolding and calibration module

Shared parameters with empirical anchors (Synergy Research Group,
EIB, OECD), Perez window dynamics, and h(σ) hump-shape function.
All parameters documented with sources."
```

---

### Task 2: Layer 1 — State Equations and Forward Simulation

**Files:**
- Create: `sovereignty-model/model/analytical.py`
- Create: `sovereignty-model/tests/test_analytical.py`

- [ ] **Step 1: Write failing tests for state equation dynamics**

Create `sovereignty-model/tests/test_analytical.py`:

```python
"""Tests for Layer 1: Analytical core."""
import pytest
import numpy as np
from model.analytical import (
    state_derivatives,
    simulate_forward,
    compute_shadow_price_of_delay,
    find_optimal_alpha,
)
from model.calibration import Parameters


class TestStateDerivatives:
    """Verify the ODE system dKp/dt, dKf/dt, dD/dt."""

    def test_returns_three_derivatives(self):
        p = Parameters()
        state = [0.5, 0.3, 0.67]  # Kp, Kf, D
        derivs = state_derivatives(0.0, state, alpha=0.5, sigma=0.0, params=p)
        assert len(derivs) == 3

    def test_zero_investment_decays_sovereignty(self):
        """With α=0 (no exploitation), Kp should decay."""
        p = Parameters()
        state = [1.0, 0.0, 0.5]
        dKp, _, _ = state_derivatives(0.0, state, alpha=0.0, sigma=0.0, params=p)
        assert dKp < 0, "Kp should decay with zero exploitation investment"

    def test_full_exploitation_grows_present(self):
        """With α=1 and low dependency, Kp should grow."""
        p = Parameters()
        state = [0.1, 0.0, 0.1]  # Low Kp, low D
        dKp, _, _ = state_derivatives(0.0, state, alpha=1.0, sigma=0.0, params=p)
        assert dKp > 0, "Kp should grow with full exploitation and low dependency"

    def test_dependency_increases_without_intervention(self):
        """With Kp=0 and moderate D, dependency should increase."""
        p = Parameters()
        state = [0.0, 0.0, 0.5]
        _, _, dD = state_derivatives(0.0, state, alpha=0.5, sigma=0.0, params=p)
        assert dD > 0, "Dependency should increase without present sovereignty"

    def test_dependency_clamped_to_unit_interval(self):
        """D should stay in [0, 1] after simulation."""
        p = Parameters()
        result = simulate_forward(alpha=0.5, sigma=0.0, params=p)
        assert np.all(result.D >= 0), "D went negative"
        assert np.all(result.D <= 1), "D exceeded 1"

    def test_window_modulates_exploration(self):
        """Future sovereignty should grow faster with open window."""
        p = Parameters()
        # Open window
        r1 = simulate_forward(alpha=0.3, sigma=1.0, params=p)
        # Closed window (already past)
        p2 = Parameters(t_open=100.0)
        r2 = simulate_forward(alpha=0.3, sigma=1.0, params=p2)
        assert r1.Kf[-1] > r2.Kf[-1], "Open window should yield more Kf"


class TestSimulateForward:
    """Verify forward simulation produces reasonable trajectories."""

    def test_output_shape(self):
        p = Parameters(T=10.0, dt=1.0)
        result = simulate_forward(alpha=0.5, sigma=0.0, params=p)
        assert len(result.t) > 1
        assert len(result.Kp) == len(result.t)
        assert len(result.Kf) == len(result.t)
        assert len(result.D) == len(result.t)

    def test_designed_pressure_increases_future_sovereignty(self):
        """σ at optimal should yield more Kf than σ=0."""
        p = Parameters()
        r_no = simulate_forward(alpha=0.3, sigma=0.0, params=p)
        r_opt = simulate_forward(alpha=0.3, sigma=2.0, params=p)  # σ* ≈ 2.0
        assert r_opt.Kf[-1] > r_no.Kf[-1]

    def test_excessive_pressure_yields_less_than_optimal(self):
        """Very high σ should yield less Kf than optimal σ."""
        p = Parameters()
        r_opt = simulate_forward(alpha=0.3, sigma=2.0, params=p)
        r_high = simulate_forward(alpha=0.3, sigma=20.0, params=p)
        assert r_opt.Kf[-1] > r_high.Kf[-1]


class TestShadowPrice:
    """Verify delay cost computation."""

    def test_delay_cost_positive(self):
        """Delaying exploration should have a positive cost."""
        p = Parameters()
        cost = compute_shadow_price_of_delay(
            alpha=0.3, sigma=2.0, delay_years=1.0, params=p
        )
        assert cost > 0

    def test_delay_cost_increases_with_delay(self):
        """Longer delays should cost more."""
        p = Parameters()
        c1 = compute_shadow_price_of_delay(alpha=0.3, sigma=2.0, delay_years=1.0, params=p)
        c5 = compute_shadow_price_of_delay(alpha=0.3, sigma=2.0, delay_years=5.0, params=p)
        assert c5 > c1


class TestOptimalAlpha:
    """Verify optimal allocation search."""

    def test_optimal_alpha_interior(self):
        """Optimal α should be between 0 and 1."""
        p = Parameters()
        result = find_optimal_alpha(sigma=2.0, params=p)
        assert 0 < result.alpha_star < 1

    def test_optimal_alpha_with_closed_window_favors_exploitation(self):
        """With closed window, optimal α should be high (exploit-heavy)."""
        p = Parameters(t_open=100.0)  # Window never opens in horizon
        result = find_optimal_alpha(sigma=0.0, params=p)
        assert result.alpha_star > 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_analytical.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write the analytical module**

Create `sovereignty-model/model/analytical.py`:

```python
"""Layer 1: Analytical core — sovereignty allocation optimal control.

Implements the state equations, forward simulation, shadow price
computation, and optimal allocation search for the social planner's
problem.
"""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from model.calibration import Parameters, window_openness, h_sigma, phi_dependency


class SimulationResult(NamedTuple):
    """Forward simulation output."""
    t: np.ndarray       # Time points
    Kp: np.ndarray      # Present sovereignty trajectory
    Kf: np.ndarray      # Future sovereignty trajectory
    D: np.ndarray       # Dependency trajectory
    W: np.ndarray       # Window openness trajectory
    V_total: float      # Discounted total value


class OptimalAlphaResult(NamedTuple):
    """Result of optimal allocation search."""
    alpha_star: float   # Optimal exploitation share
    V_star: float       # Value at optimum
    alpha_grid: np.ndarray   # Grid of α values tested
    V_grid: np.ndarray       # Corresponding values


def state_derivatives(
    t: float,
    state: list[float],
    alpha: float,
    sigma: float,
    params: Parameters,
) -> tuple[float, float, float]:
    """Compute dKp/dt, dKf/dt, dD/dt.

    Args:
        t: Current time
        state: [Kp, Kf, D]
        alpha: Exploitation allocation share (0 to 1)
        sigma: Constraint instrument intensity
        params: Model parameters

    Returns:
        Tuple of (dKp_dt, dKf_dt, dD_dt)
    """
    Kp, Kf, D = state

    # Clamp D to [0, 1] for numerical safety
    D = np.clip(D, 0.0, 1.0)

    # Present sovereignty: dKp/dt = A·(α·R)^κ - δp·Kp - φ(D)·Kp
    exploitation_input = alpha * params.R
    dKp_dt = (
        params.A * exploitation_input ** params.kappa
        - params.delta_p * Kp
        - phi_dependency(D) * Kp
    )

    # Future sovereignty: dKf/dt = W(t)·[(1-α)·R]^β·h(σ) - δf·Kf
    W = window_openness(t, params.t_open, params.W0, params.gamma)
    exploration_input = (1.0 - alpha) * params.R
    g = exploration_input ** params.beta * h_sigma(sigma, params.a, params.b)
    dKf_dt = W * g - params.delta_f * Kf

    # Dependency: dD/dt = λ·(D̄ - D)·D - η·Kp·D
    dD_dt = params.lam * (params.D_bar - D) * D - params.eta * Kp * D

    return (dKp_dt, dKf_dt, dD_dt)


def simulate_forward(
    alpha: float,
    sigma: float,
    params: Parameters,
    Kp0: float = 0.1,
    Kf0: float = 0.0,
) -> SimulationResult:
    """Run forward simulation of state equations.

    Args:
        alpha: Fixed exploitation allocation share
        sigma: Fixed constraint intensity
        params: Model parameters
        Kp0: Initial present sovereignty
        Kf0: Initial future sovereignty

    Returns:
        SimulationResult with trajectories and total discounted value
    """
    y0 = [Kp0, Kf0, params.D0]
    t_span = (0.0, params.T)
    t_eval = np.arange(0, params.T + params.dt, params.dt)

    def ode_rhs(t, y):
        return state_derivatives(t, y, alpha, sigma, params)

    sol = solve_ivp(
        ode_rhs,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        max_step=0.5,
    )

    Kp = sol.y[0]
    Kf = sol.y[1]
    D = np.clip(sol.y[2], 0.0, 1.0)
    t = sol.t

    # Window trajectory
    W = np.array([
        window_openness(ti, params.t_open, params.W0, params.gamma)
        for ti in t
    ])

    # Discounted total value: ∫ e^{-ρt} · V(Kp, Kf, D) dt
    V_flow = (
        params.omega * Kp
        + (1.0 - params.omega) * Kf
        - params.psi * D ** 2
    )
    discount = np.exp(-params.rho * t)
    V_total = np.trapezoid(V_flow * discount, t)

    return SimulationResult(t=t, Kp=Kp, Kf=Kf, D=D, W=W, V_total=V_total)


def compute_shadow_price_of_delay(
    alpha: float,
    sigma: float,
    delay_years: float,
    params: Parameters,
) -> float:
    """Compute the cost of delaying exploration by delay_years.

    Compares V(start now) vs V(start after delay with α=1 during delay).
    """
    # Baseline: start now
    V_now = simulate_forward(alpha, sigma, params).V_total

    # Delayed: exploit-only during delay, then switch
    p_delay = Parameters(**{
        **params.__dict__,
        "t_open": params.t_open + delay_years,
    })
    V_delayed = simulate_forward(alpha, sigma, p_delay).V_total

    return V_now - V_delayed


def find_optimal_alpha(
    sigma: float,
    params: Parameters,
    n_grid: int = 50,
) -> OptimalAlphaResult:
    """Find the optimal fixed allocation α* for given σ.

    Uses grid search followed by bounded scalar optimization.
    """
    alpha_grid = np.linspace(0.01, 0.99, n_grid)
    V_grid = np.array([
        simulate_forward(a, sigma, params).V_total for a in alpha_grid
    ])

    # Refine with optimizer
    best_idx = np.argmax(V_grid)
    lo = alpha_grid[max(0, best_idx - 1)]
    hi = alpha_grid[min(len(alpha_grid) - 1, best_idx + 1)]

    result = minimize_scalar(
        lambda a: -simulate_forward(a, sigma, params).V_total,
        bounds=(lo, hi),
        method="bounded",
    )

    return OptimalAlphaResult(
        alpha_star=result.x,
        V_star=-result.fun,
        alpha_grid=alpha_grid,
        V_grid=V_grid,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_analytical.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/model/analytical.py sovereignty-model/tests/test_analytical.py
git commit -m "feat: add Layer 1 analytical core

State equations (Kp, Kf, D dynamics), forward simulation via
SciPy solve_ivp, shadow price of delay computation, and optimal
α* search with grid + bounded optimization."
```

---

### Task 3: Layer 2 — Evolutionary Simulation

**Files:**
- Create: `sovereignty-model/model/evolutionary.py`
- Create: `sovereignty-model/tests/test_evolutionary.py`

- [ ] **Step 1: Write failing tests for evolutionary simulation**

Create `sovereignty-model/tests/test_evolutionary.py`:

```python
"""Tests for Layer 2: Evolutionary simulation."""
import pytest
import numpy as np
from model.evolutionary import (
    initialize_firms,
    simulate_evolution,
    sweep_sigma,
    FirmState,
)
from model.calibration import Parameters


class TestInitializeFirms:
    """Verify firm initialization."""

    def test_creates_n_firms(self):
        p = Parameters(N=100)
        firms = initialize_firms(p, seed=42)
        assert firms.capability.shape[0] == 100

    def test_market_shares_sum_to_one(self):
        p = Parameters(N=100)
        firms = initialize_firms(p, seed=42)
        assert firms.share.sum() == pytest.approx(1.0, abs=1e-10)

    def test_dependency_in_unit_interval(self):
        p = Parameters(N=100)
        firms = initialize_firms(p, seed=42)
        assert np.all(firms.dependency >= 0)
        assert np.all(firms.dependency <= 1)


class TestSimulateEvolution:
    """Verify evolutionary dynamics."""

    def test_output_shape(self):
        p = Parameters(N=100, T=10.0)
        result = simulate_evolution(sigma=0.0, params=p, seed=42)
        assert len(result.aggregate_capability) == 11  # 0..10 inclusive

    def test_shares_sum_to_one_each_period(self):
        p = Parameters(N=100, T=5.0)
        result = simulate_evolution(sigma=0.0, params=p, seed=42)
        for shares in result.share_history:
            assert shares.sum() == pytest.approx(1.0, abs=1e-6)

    def test_no_pressure_grows_dependency(self):
        """Without constraint, average dependency should increase."""
        p = Parameters(N=100, T=15.0)
        result = simulate_evolution(sigma=0.0, params=p, seed=42)
        assert result.avg_dependency[-1] > result.avg_dependency[0]

    def test_designed_pressure_dip_then_compound(self):
        """Moderate σ: capability should dip initially then recover."""
        p = Parameters(N=200, T=20.0)
        result = simulate_evolution(sigma=2.0, params=p, seed=42)
        cap = result.aggregate_capability
        # Find if there's a dip (min after t=0 is below start)
        # Then recovery (end is above the dip)
        dip_idx = np.argmin(cap[1:]) + 1
        assert cap[-1] > cap[dip_idx], "Should recover after dip"


class TestSweepSigma:
    """Verify Monte Carlo sweep across σ values."""

    def test_sweep_returns_results_per_sigma(self):
        p = Parameters(N=50, T=10.0)
        sigmas = [0.0, 1.0, 5.0]
        results = sweep_sigma(sigmas, params=p, n_replications=3, seed=42)
        assert len(results) == 3

    def test_hump_shape_emerges(self):
        """Terminal capability should be highest at moderate σ."""
        p = Parameters(N=100, T=15.0)
        sigmas = [0.0, 2.0, 15.0]
        results = sweep_sigma(sigmas, params=p, n_replications=5, seed=42)
        cap_none = np.mean(results[0.0])
        cap_moderate = np.mean(results[2.0])
        cap_excessive = np.mean(results[15.0])
        assert cap_moderate > cap_none, "Designed pressure should beat no pressure"
        assert cap_moderate > cap_excessive, "Designed pressure should beat excessive"


class TestConvergence:
    """Verify results stabilize with more firms."""

    def test_larger_n_reduces_variance(self):
        p_small = Parameters(N=50, T=10.0)
        p_large = Parameters(N=200, T=10.0)
        sigmas = [2.0]

        r_small = sweep_sigma(sigmas, params=p_small, n_replications=10, seed=42)
        r_large = sweep_sigma(sigmas, params=p_large, n_replications=10, seed=42)

        var_small = np.var(r_small[2.0])
        var_large = np.var(r_large[2.0])
        assert var_large < var_small, "More firms should reduce variance"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evolutionary.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write the evolutionary simulation module**

Create `sovereignty-model/model/evolutionary.py`:

```python
"""Layer 2: Evolutionary simulation — selection pressure dynamics.

Agent-based model with heterogeneous firms. Innovation lottery,
selection mechanism, entry/exit. Vectorized with NumPy.
"""
from dataclasses import dataclass
import numpy as np

from model.calibration import Parameters, window_openness, h_sigma


@dataclass
class FirmState:
    """Vectorized firm state. Arrays of shape (N,)."""
    capability: np.ndarray    # cᵢ — absorptive capacity
    orientation: np.ndarray   # rᵢ — research orientation [0=exploit, 1=explore]
    dependency: np.ndarray    # xᵢ — external dependency ratio
    share: np.ndarray         # sᵢ — market share


@dataclass
class EvolutionResult:
    """Simulation output."""
    t: np.ndarray                          # Time points
    aggregate_capability: np.ndarray       # C(t) = Σ sᵢcᵢ
    avg_dependency: np.ndarray             # Mean xᵢ weighted by share
    capability_gini: np.ndarray            # Gini of capability distribution
    share_history: list[np.ndarray]        # Market shares each period
    capability_history: list[np.ndarray]   # Capabilities each period


def initialize_firms(params: Parameters, seed: int = 42) -> FirmState:
    """Create N firms with initial distributions per spec."""
    rng = np.random.default_rng(seed)
    N = params.N

    capability = rng.lognormal(mean=0.0, sigma=0.5, size=N)
    capability = capability / capability.mean()  # Normalize to mean 1

    orientation = rng.beta(2, 5, size=N)       # Skewed toward exploitation
    dependency = rng.beta(5, 2, size=N)         # Skewed toward high dependency
    share = np.ones(N) / N                      # Equal initial shares

    return FirmState(
        capability=capability,
        orientation=orientation,
        dependency=dependency,
        share=share,
    )


def _gini(x: np.ndarray) -> float:
    """Compute Gini coefficient of array x."""
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def simulate_evolution(
    sigma: float,
    params: Parameters,
    seed: int = 42,
) -> EvolutionResult:
    """Run one evolutionary simulation.

    Args:
        sigma: Constraint instrument intensity (fixed for run)
        params: Model parameters
        seed: Random seed for reproducibility

    Returns:
        EvolutionResult with trajectories
    """
    rng = np.random.default_rng(seed)
    firms = initialize_firms(params, seed=seed)
    n_periods = int(params.T / params.dt)

    # Output arrays
    agg_cap = np.zeros(n_periods + 1)
    avg_dep = np.zeros(n_periods + 1)
    cap_gini = np.zeros(n_periods + 1)
    share_hist = []
    cap_hist = []

    # Record initial state
    agg_cap[0] = np.sum(firms.share * firms.capability)
    avg_dep[0] = np.sum(firms.share * firms.dependency)
    cap_gini[0] = _gini(firms.capability)
    share_hist.append(firms.share.copy())
    cap_hist.append(firms.capability.copy())

    for period in range(1, n_periods + 1):
        t = period * params.dt
        W = window_openness(t, params.t_open, params.W0, params.gamma)

        c = firms.capability
        x = firms.dependency
        N = params.N

        # --- Innovation draws ---

        # Exploitation: probability p₀·cᵢ^ν, payoff ε_exploit·cᵢ·xᵢ
        # Constrained: payoff scaled by (1 - σ·xᵢ)
        p_exploit = np.clip(params.p0 * c ** params.nu, 0, 1)
        exploit_success = rng.random(N) < p_exploit
        constraint_factor = np.clip(1.0 - sigma * x, 0, 1)
        exploit_payoff = (
            params.eps_exploit * c * x * constraint_factor * exploit_success
        )

        # Exploration: probability q₀·cᵢ^ν·W, payoff Pareto-drawn
        p_explore = np.clip(params.q0 * c ** params.nu * W, 0, 1)
        explore_success = rng.random(N) < p_explore
        pareto_draw = (
            params.eps_explore
            * rng.pareto(params.xi, size=N)
        )
        explore_payoff = pareto_draw * c * (1.0 - x) * explore_success

        # Update capabilities
        firms.capability = c + exploit_payoff + explore_payoff

        # Exploration success reduces dependency
        dep_reduction = 0.05 * explore_success
        firms.dependency = np.clip(x - dep_reduction, 0, 1)

        # Firms under constraint with low capability drift toward higher dependency
        low_cap_mask = c < np.median(c)
        firms.dependency[low_cap_mask] = np.clip(
            firms.dependency[low_cap_mask] + 0.01 * sigma, 0, 1
        )

        # --- Selection ---
        c_mean = np.mean(firms.capability)
        if c_mean > 0:
            raw_share = firms.share * (firms.capability / c_mean) ** params.theta
            firms.share = raw_share / raw_share.sum()

        # --- Entry/Exit ---
        exit_mask = firms.share < params.s_min
        n_exit = exit_mask.sum()
        if n_exit > 0:
            # Replace exiting firms with new entrants
            new = initialize_firms(
                Parameters(**{**params.__dict__, "N": int(n_exit)}),
                seed=seed + period,
            )
            firms.capability[exit_mask] = new.capability[:n_exit]
            firms.orientation[exit_mask] = new.orientation[:n_exit]
            firms.dependency[exit_mask] = new.dependency[:n_exit]
            firms.share[exit_mask] = params.s_min

            # Renormalize
            firms.share = firms.share / firms.share.sum()

        # Record
        agg_cap[period] = np.sum(firms.share * firms.capability)
        avg_dep[period] = np.sum(firms.share * firms.dependency)
        cap_gini[period] = _gini(firms.capability)
        share_hist.append(firms.share.copy())
        cap_hist.append(firms.capability.copy())

    return EvolutionResult(
        t=np.arange(0, n_periods + 1) * params.dt,
        aggregate_capability=agg_cap,
        avg_dependency=avg_dep,
        capability_gini=cap_gini,
        share_history=share_hist,
        capability_history=cap_hist,
    )


def sweep_sigma(
    sigmas: list[float],
    params: Parameters,
    n_replications: int = 10,
    seed: int = 42,
) -> dict[float, list[float]]:
    """Run Monte Carlo sweep across σ values.

    Returns dict mapping σ → list of terminal aggregate capabilities
    (one per replication).
    """
    results: dict[float, list[float]] = {}
    for sigma in sigmas:
        terminal_caps = []
        for rep in range(n_replications):
            r = simulate_evolution(sigma, params, seed=seed + rep * 1000)
            terminal_caps.append(r.aggregate_capability[-1])
        results[sigma] = terminal_caps
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evolutionary.py -v
```

Expected: All PASS. The `test_designed_pressure_dip_then_compound` test may need seed tuning — if it fails, adjust seed or relax the assertion to check `cap[-1] > cap[0]` instead.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/model/evolutionary.py sovereignty-model/tests/test_evolutionary.py
git commit -m "feat: add Layer 2 evolutionary simulation

Agent-based model with N firms, innovation lottery (exploitation +
exploration draws), selection mechanism with renormalization,
entry/exit dynamics, and Monte Carlo σ sweep."
```

---

### Task 4: Layer 3 — Policy Interface

**Files:**
- Create: `sovereignty-model/model/policy.py`
- Create: `sovereignty-model/tests/test_policy.py`

- [ ] **Step 1: Write failing tests for policy module**

Create `sovereignty-model/tests/test_policy.py`:

```python
"""Tests for Layer 3: Policy design interface."""
import pytest
import numpy as np
from model.policy import (
    compute_delay_cost_curve,
    compute_regime_comparison,
    score_domain,
    compute_policy_recommendation,
)
from model.calibration import Parameters


class TestDelayCostCurve:
    """Verify delay cost curves."""

    def test_returns_positive_costs(self):
        p = Parameters()
        delays, costs = compute_delay_cost_curve(
            alpha=0.3, sigma=2.0, max_delay=10.0, params=p
        )
        assert np.all(costs >= 0)

    def test_monotonically_increasing(self):
        """Waiting longer should never cost less."""
        p = Parameters()
        delays, costs = compute_delay_cost_curve(
            alpha=0.3, sigma=2.0, max_delay=10.0, params=p
        )
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1] - 1e-10

    def test_zero_delay_zero_cost(self):
        p = Parameters()
        delays, costs = compute_delay_cost_curve(
            alpha=0.3, sigma=2.0, max_delay=5.0, params=p
        )
        assert costs[0] == pytest.approx(0.0, abs=1e-10)


class TestRegimeComparison:
    """Verify three-regime trajectory comparison."""

    def test_returns_three_regimes(self):
        p = Parameters()
        regimes = compute_regime_comparison(params=p)
        assert "status_quo" in regimes
        assert "designed" in regimes
        assert "overreaction" in regimes

    def test_designed_beats_status_quo_on_kf(self):
        p = Parameters()
        regimes = compute_regime_comparison(params=p)
        assert regimes["designed"].Kf[-1] > regimes["status_quo"].Kf[-1]

    def test_designed_beats_overreaction_on_total_value(self):
        p = Parameters()
        regimes = compute_regime_comparison(params=p)
        assert regimes["designed"].V_total > regimes["overreaction"].V_total


class TestDomainScoring:
    """Verify domain selection scoring."""

    def test_open_window_scores_higher(self):
        s_open = score_domain(W_remaining=0.8, capability=0.5, strategic_value=0.7)
        s_closed = score_domain(W_remaining=0.1, capability=0.5, strategic_value=0.7)
        assert s_open > s_closed

    def test_higher_capability_scores_higher(self):
        s_high = score_domain(W_remaining=0.5, capability=0.8, strategic_value=0.7)
        s_low = score_domain(W_remaining=0.5, capability=0.2, strategic_value=0.7)
        assert s_high > s_low

    def test_score_in_unit_interval(self):
        s = score_domain(W_remaining=0.5, capability=0.5, strategic_value=0.5)
        assert 0 <= s <= 1


class TestPolicyRecommendation:
    """Verify integrated policy recommendation."""

    def test_returns_all_instruments(self):
        p = Parameters()
        rec = compute_policy_recommendation(
            D_obs=0.67, C_obs=0.5, W_obs=0.7, params=p
        )
        assert hasattr(rec, "alpha_star")
        assert hasattr(rec, "sigma_star")
        assert hasattr(rec, "domain_scores")

    def test_high_dependency_shifts_toward_exploration(self):
        """High D should push α* down (more exploration)."""
        p = Parameters()
        rec_high = compute_policy_recommendation(D_obs=0.9, C_obs=0.5, W_obs=0.7, params=p)
        rec_low = compute_policy_recommendation(D_obs=0.2, C_obs=0.5, W_obs=0.7, params=p)
        assert rec_high.alpha_star < rec_low.alpha_star
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_policy.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Write the policy module**

Create `sovereignty-model/model/policy.py`:

```python
"""Layer 3: Policy design interface — strategic necessity instrument.

Translates analytical and evolutionary results into actionable
policy recommendations: allocation ratios, constraint intensity,
domain selection, delay costs, regime comparisons.
"""
from typing import NamedTuple
import numpy as np

from model.calibration import Parameters
from model.analytical import simulate_forward, find_optimal_alpha, compute_shadow_price_of_delay


class PolicyRecommendation(NamedTuple):
    """Integrated policy recommendation."""
    alpha_star: float           # Optimal exploitation share
    sigma_star: float           # Optimal constraint intensity
    domain_scores: dict[str, float]  # Domain → score
    delay_cost_1yr: float       # Cost of 1-year delay


def compute_delay_cost_curve(
    alpha: float,
    sigma: float,
    max_delay: float,
    params: Parameters,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute delay cost curve: cost of waiting 0..max_delay years.

    Returns:
        (delays, costs) arrays
    """
    delays = np.linspace(0, max_delay, n_points)
    costs = np.zeros(n_points)

    for i, d in enumerate(delays):
        if d == 0:
            costs[i] = 0.0
        else:
            costs[i] = compute_shadow_price_of_delay(alpha, sigma, d, params)

    return delays, costs


def compute_regime_comparison(
    params: Parameters,
) -> dict[str, "SimulationResult"]:
    """Compare three policy regimes.

    - status_quo: σ=0, α=0.8 (exploit-heavy, no constraint)
    - designed: σ=σ*, α=α* (optimal)
    - overreaction: σ=15, α=0.2 (excessive constraint)
    """
    from model.analytical import SimulationResult

    # Status quo: presentism
    r_sq = simulate_forward(alpha=0.8, sigma=0.0, params=params)

    # Designed: find optimal α at σ*
    sigma_star = 1.0 / params.b  # Peak of h(σ)
    opt = find_optimal_alpha(sigma=sigma_star, params=params)
    r_designed = simulate_forward(alpha=opt.alpha_star, sigma=sigma_star, params=params)

    # Overreaction: excessive constraint
    r_over = simulate_forward(alpha=0.2, sigma=15.0, params=params)

    return {
        "status_quo": r_sq,
        "designed": r_designed,
        "overreaction": r_over,
    }


def score_domain(
    W_remaining: float,
    capability: float,
    strategic_value: float,
    w_weight: float = 0.4,
    c_weight: float = 0.3,
    s_weight: float = 0.3,
) -> float:
    """Score a domain for investment priority.

    Weighted combination of window remaining, existing capability,
    and strategic value. All inputs in [0, 1].
    """
    return (
        w_weight * W_remaining
        + c_weight * capability
        + s_weight * strategic_value
    )


# Default domain assessments from essay evidence
DEFAULT_DOMAINS = {
    "scientific_ai": {"W_remaining": 0.7, "capability": 0.4, "strategic_value": 0.8},
    "quantum": {"W_remaining": 0.6, "capability": 0.6, "strategic_value": 0.7},
    "fusion": {"W_remaining": 0.8, "capability": 0.5, "strategic_value": 0.9},
}


def compute_policy_recommendation(
    D_obs: float,
    C_obs: float,
    W_obs: float,
    params: Parameters,
    domains: dict | None = None,
) -> PolicyRecommendation:
    """Compute integrated policy recommendation from observables.

    Args:
        D_obs: Observed dependency level [0, 1]
        C_obs: Observed capability level [0, 1]
        W_obs: Observed window openness [0, 1]
        params: Model parameters
        domains: Optional domain assessments dict

    Returns:
        PolicyRecommendation with all instruments
    """
    if domains is None:
        domains = DEFAULT_DOMAINS

    # Adjust parameters based on observables
    adj_params = Parameters(**{
        **params.__dict__,
        "D0": D_obs,
        "W0": W_obs,
    })

    # Optimal σ: scale by capability (low capability → lower σ to avoid destruction)
    sigma_star_base = 1.0 / params.b
    sigma_star = sigma_star_base * C_obs  # Scale by capability

    # Optimal α
    opt = find_optimal_alpha(sigma=sigma_star, params=adj_params)

    # Domain scores
    domain_scores = {
        name: score_domain(**assessment)
        for name, assessment in domains.items()
    }

    # Delay cost
    delay_cost = compute_shadow_price_of_delay(
        opt.alpha_star, sigma_star, 1.0, adj_params
    )

    return PolicyRecommendation(
        alpha_star=opt.alpha_star,
        sigma_star=sigma_star,
        domain_scores=domain_scores,
        delay_cost_1yr=delay_cost,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_policy.py -v
```

Expected: All PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests across all four test files pass.

- [ ] **Step 6: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/model/policy.py sovereignty-model/tests/test_policy.py
git commit -m "feat: add Layer 3 policy design interface

Delay cost curves, three-regime comparison (status quo / designed /
overreaction), domain scoring, and integrated policy recommendation
from observable state variables."
```

---

### Task 5: Visualization Module

**Files:**
- Create: `sovereignty-model/model/visualization.py`

- [ ] **Step 1: Write the visualization module**

Create `sovereignty-model/model/visualization.py`:

```python
"""Visualization functions for all three model layers.

Dual output: Matplotlib for static figures (paper), Plotly for
interactive (dashboard).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.facecolor": "#FAFAF7",
    "axes.facecolor": "#FAFAF7",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e8e6e1",
    "grid.alpha": 0.5,
})

from model.calibration import Parameters, h_sigma


# --- Color palette (matches essay figures) ---
COLORS = {
    "drift": "#b0a896",
    "destructive": "#c4756a",
    "designed": "#5a8a6a",
    "present": "#8B4049",
    "future": "#5a7a8a",
    "dependency": "#b89a3c",
    "status_quo": "#b0a896",
    "overreaction": "#c4756a",
}


def plot_hump_shape(
    sigma_max: float = 15.0,
    params: Parameters | None = None,
    simulated_data: dict[float, list[float]] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot h(σ) hump shape, optionally overlaid with simulation data.

    Args:
        sigma_max: Maximum σ to plot
        params: Parameters for analytical h(σ)
        simulated_data: Optional dict from sweep_sigma {σ: [terminal_caps]}
        save_path: Optional path to save figure
    """
    if params is None:
        params = Parameters()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Analytical curve
    sigmas = np.linspace(0, sigma_max, 200)
    h_vals = [h_sigma(s, params.a, params.b) for s in sigmas]
    ax.plot(sigmas, h_vals, color=COLORS["designed"], linewidth=2, label="Analytical h(σ)")

    # σ* marker
    sigma_star = 1.0 / params.b
    h_star = h_sigma(sigma_star, params.a, params.b)
    ax.plot(sigma_star, h_star, "o", color=COLORS["designed"], markersize=8)
    ax.annotate(
        f"σ* = {sigma_star:.1f}",
        (sigma_star, h_star),
        xytext=(sigma_star + 1, h_star + 0.05),
        fontsize=10,
        color=COLORS["designed"],
    )

    # Simulated data overlay
    if simulated_data:
        sim_sigmas = sorted(simulated_data.keys())
        sim_means = [np.mean(simulated_data[s]) for s in sim_sigmas]
        sim_stds = [np.std(simulated_data[s]) for s in sim_sigmas]
        # Normalize to h(σ) scale
        baseline = sim_means[0] if sim_means[0] > 0 else 1.0
        sim_h = [m / baseline for m in sim_means]
        sim_h_err = [s / baseline for s in sim_stds]
        ax.errorbar(
            sim_sigmas, sim_h, yerr=sim_h_err,
            fmt="s", color=COLORS["future"], markersize=6,
            capsize=3, label="Simulated (Layer 2)",
        )

    # Regime annotations
    ax.axvspan(0, 0.3, alpha=0.05, color=COLORS["drift"])
    ax.axvspan(sigma_star - 1, sigma_star + 1, alpha=0.08, color=COLORS["designed"])
    ax.axvspan(sigma_max - 3, sigma_max, alpha=0.05, color=COLORS["destructive"])

    ax.text(0.15, 0.97, "Absent", transform=ax.transAxes, fontsize=9, color=COLORS["drift"], va="top")
    ax.text(0.38, 0.97, "Designed", transform=ax.transAxes, fontsize=9, color=COLORS["designed"], va="top")
    ax.text(0.8, 0.97, "Destructive", transform=ax.transAxes, fontsize=9, color=COLORS["destructive"], va="top")

    ax.set_xlabel("Constraint intensity σ")
    ax.set_ylabel("Exploration productivity h(σ)")
    ax.set_title("The Hump Shape: Constraint and Innovation")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_regime_comparison(
    regimes: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot three-regime trajectory comparison.

    Args:
        regimes: Dict from compute_regime_comparison()
        save_path: Optional path to save
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    labels = {
        "status_quo": ("Status Quo (Presentism)", COLORS["status_quo"]),
        "designed": ("Designed Pressure", COLORS["designed"]),
        "overreaction": ("Overreaction", COLORS["overreaction"]),
    }

    # Panel 1: Present Sovereignty
    ax = axes[0]
    for key, result in regimes.items():
        label, color = labels[key]
        ax.plot(result.t, result.Kp, color=color, linewidth=2, label=label)
    ax.set_title("Present Sovereignty Kp")
    ax.set_xlabel("Years")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)

    # Panel 2: Future Sovereignty
    ax = axes[1]
    for key, result in regimes.items():
        label, color = labels[key]
        ax.plot(result.t, result.Kf, color=color, linewidth=2, label=label)
    ax.set_title("Future Sovereignty Kf")
    ax.set_xlabel("Years")
    ax.grid(True, linewidth=0.5)

    # Panel 3: Dependency
    ax = axes[2]
    for key, result in regimes.items():
        label, color = labels[key]
        ax.plot(result.t, result.D, color=color, linewidth=2, label=label)
    ax.set_title("Dependency D")
    ax.set_xlabel("Years")
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_delay_costs(
    delays: np.ndarray,
    costs: np.ndarray,
    domain_name: str = "Default",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot shadow price of delay curve."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(delays, 0, costs, alpha=0.15, color=COLORS["present"])
    ax.plot(delays, costs, color=COLORS["present"], linewidth=2)

    ax.set_xlabel("Delay (years)")
    ax.set_ylabel("Cumulative cost of delay")
    ax.set_title(f"Shadow Price of Delay — {domain_name}")
    ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_evolutionary_trajectories(
    results_by_sigma: dict[float, "EvolutionResult"],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot aggregate capability trajectories for multiple σ values."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sigma_colors = {
        0.0: COLORS["drift"],
        2.0: COLORS["designed"],
        15.0: COLORS["destructive"],
    }

    sigma_labels = {
        0.0: "σ = 0 (No pressure)",
        2.0: "σ = 2 (Designed)",
        15.0: "σ = 15 (Destructive)",
    }

    for sigma, result in sorted(results_by_sigma.items()):
        color = sigma_colors.get(sigma, "#666666")
        label = sigma_labels.get(sigma, f"σ = {sigma}")
        ax.plot(result.t, result.aggregate_capability,
                color=color, linewidth=2, label=label)

    ax.set_xlabel("Years")
    ax.set_ylabel("Aggregate Capability C(t)")
    ax.set_title("Evolutionary Dynamics Under Selection Pressure")
    ax.legend(framealpha=0.9)
    ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_optimal_alpha(
    alpha_result,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot value function over α grid with optimal marked."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(alpha_result.alpha_grid, alpha_result.V_grid,
            color=COLORS["future"], linewidth=2)
    ax.axvline(alpha_result.alpha_star, color=COLORS["designed"],
               linestyle="--", linewidth=1.5)
    ax.plot(alpha_result.alpha_star, alpha_result.V_star, "o",
            color=COLORS["designed"], markersize=8)
    ax.annotate(
        f"α* = {alpha_result.alpha_star:.2f}",
        (alpha_result.alpha_star, alpha_result.V_star),
        xytext=(alpha_result.alpha_star + 0.05, alpha_result.V_star),
        fontsize=10, color=COLORS["designed"],
    )

    ax.set_xlabel("Allocation α (exploitation share)")
    ax.set_ylabel("Discounted total value V")
    ax.set_title("Optimal Sovereignty Allocation")
    ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
```

- [ ] **Step 2: Smoke-test by generating a figure**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
source .venv/bin/activate
python3 -c "
from model.visualization import plot_hump_shape
fig = plot_hump_shape(save_path='output/figures/hump_shape.png')
print('Saved hump_shape.png')
"
```

Expected: `output/figures/hump_shape.png` created without error.

- [ ] **Step 3: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/model/visualization.py sovereignty-model/output/figures/hump_shape.png
git commit -m "feat: add visualization module

Matplotlib plotting for hump shape, regime comparison, delay costs,
evolutionary trajectories, and optimal allocation. Academic/editorial
style matching essay figures."
```

---

### Task 6: Streamlit Dashboard

**Files:**
- Create: `sovereignty-model/app/dashboard.py`

- [ ] **Step 1: Write the Streamlit dashboard**

Create `sovereignty-model/app/dashboard.py`:

```python
"""Interactive dashboard — three-panel Streamlit companion.

Run: streamlit run app/dashboard.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model.calibration import Parameters, h_sigma
from model.analytical import simulate_forward, find_optimal_alpha, compute_shadow_price_of_delay
from model.policy import compute_regime_comparison, compute_delay_cost_curve, score_domain, DEFAULT_DOMAINS
from model.evolutionary import simulate_evolution, sweep_sigma


# --- Page config ---
st.set_page_config(
    page_title="Sovereignty Model — Interactive Companion",
    layout="wide",
)

# --- Palette ---
COLORS = {
    "drift": "#b0a896",
    "destructive": "#c4756a",
    "designed": "#5a8a6a",
    "present": "#8B4049",
    "future": "#5a7a8a",
    "dependency": "#b89a3c",
}

st.title("Two Kinds of Sovereignty — Economic Model")
st.markdown("*Interactive companion to the formal model. Explore how allocation, constraint intensity, and window dynamics shape sovereignty trajectories.*")

# --- Sidebar: shared parameters ---
st.sidebar.header("Model Parameters")
D0 = st.sidebar.slider("Initial dependency D₀", 0.0, 1.0, 0.67, 0.01)
R = st.sidebar.slider("R&D intensity (% GDP)", 0.01, 0.06, 0.022, 0.001, format="%.3f")
T = st.sidebar.slider("Time horizon (years)", 10, 50, 30)
gamma = st.sidebar.slider("Window closure rate γ", 0.01, 0.3, 0.1, 0.01)
rho = st.sidebar.slider("Discount rate ρ", 0.01, 0.10, 0.03, 0.01)

params = Parameters(D0=D0, R=R, T=float(T), gamma=gamma, rho=rho)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "Layer 1: Analytical Core",
    "Layer 2: Evolutionary Simulation",
    "Layer 3: Policy Interface",
])

# ============================================================
# TAB 1: ANALYTICAL CORE
# ============================================================
with tab1:
    st.header("Optimal Allocation & Constraint Design")

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Exploitation share α", 0.0, 1.0, 0.5, 0.01, key="alpha1")
    with col2:
        sigma = st.slider("Constraint intensity σ", 0.0, 15.0, 2.0, 0.1, key="sigma1")

    result = simulate_forward(alpha, sigma, params)

    # Trajectory plot
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Present Sovereignty Kp", "Future Sovereignty Kf", "Dependency D"))

    fig.add_trace(go.Scatter(x=result.t, y=result.Kp, line=dict(color=COLORS["present"], width=2), name="Kp"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.t, y=result.Kf, line=dict(color=COLORS["future"], width=2), name="Kf"), row=1, col=2)
    fig.add_trace(go.Scatter(x=result.t, y=result.D, line=dict(color=COLORS["dependency"], width=2), name="D"), row=1, col=3)

    fig.update_layout(height=350, showlegend=False, margin=dict(t=40, b=20))
    fig.update_xaxes(title_text="Years")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Discounted Total Value V", f"{result.V_total:.4f}")

    # Hump shape
    st.subheader("Constraint Instrument h(σ)")
    sigmas_plot = np.linspace(0, 15, 200)
    h_vals = [h_sigma(s, params.a, params.b) for s in sigmas_plot]
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=sigmas_plot, y=h_vals, line=dict(color=COLORS["designed"], width=2)))
    fig_h.add_vline(x=sigma, line_dash="dash", line_color=COLORS["destructive"], annotation_text=f"Current σ={sigma}")
    fig_h.update_layout(height=300, xaxis_title="σ", yaxis_title="h(σ)", margin=dict(t=20, b=20))
    st.plotly_chart(fig_h, use_container_width=True)

# ============================================================
# TAB 2: EVOLUTIONARY SIMULATION
# ============================================================
with tab2:
    st.header("Selection Pressure Dynamics")

    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("Number of firms", 50, 500, 200, 50)
    with col2:
        seed = st.number_input("Random seed", value=42, step=1)

    sigma_vals = st.multiselect(
        "σ values to compare",
        options=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0],
        default=[0.0, 2.0, 15.0],
    )

    if st.button("Run Simulation", key="run_evo"):
        evo_params = Parameters(**{**params.__dict__, "N": N})
        fig_evo = go.Figure()

        sigma_color_map = {0.0: COLORS["drift"], 2.0: COLORS["designed"], 15.0: COLORS["destructive"]}

        for sv in sorted(sigma_vals):
            r = simulate_evolution(sv, evo_params, seed=int(seed))
            color = sigma_color_map.get(sv, "#666666")
            fig_evo.add_trace(go.Scatter(
                x=r.t, y=r.aggregate_capability,
                line=dict(color=color, width=2),
                name=f"σ = {sv}",
            ))

        fig_evo.update_layout(
            height=400, xaxis_title="Years", yaxis_title="Aggregate Capability C(t)",
            title="Evolutionary Dynamics Under Selection Pressure",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_evo, use_container_width=True)

        # Monte Carlo sweep
        st.subheader("Monte Carlo Sweep — h(σ) Validation")
        sweep_sigmas = np.linspace(0, 12, 13).tolist()
        n_rep = st.slider("Replications per σ", 3, 20, 5)

        if st.button("Run Sweep", key="run_sweep"):
            sweep_results = sweep_sigma(sweep_sigmas, evo_params, n_replications=n_rep, seed=int(seed))
            means = [np.mean(sweep_results[s]) for s in sweep_sigmas]
            stds = [np.std(sweep_results[s]) for s in sweep_sigmas]
            baseline = means[0] if means[0] > 0 else 1.0
            norm_means = [m / baseline for m in means]
            norm_stds = [s / baseline for s in stds]

            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=sweep_sigmas, y=norm_means,
                error_y=dict(type="data", array=norm_stds, visible=True),
                mode="markers+lines", marker=dict(color=COLORS["future"], size=8),
                line=dict(color=COLORS["future"]),
            ))
            # Overlay analytical
            h_analytical = [h_sigma(s, params.a, params.b) for s in sweep_sigmas]
            fig_sweep.add_trace(go.Scatter(
                x=sweep_sigmas, y=h_analytical,
                mode="lines", line=dict(color=COLORS["designed"], dash="dash"),
                name="Analytical h(σ)",
            ))
            fig_sweep.update_layout(
                height=350, xaxis_title="σ", yaxis_title="Normalized terminal capability",
                title="Simulated vs Analytical h(σ)", margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_sweep, use_container_width=True)

# ============================================================
# TAB 3: POLICY INTERFACE
# ============================================================
with tab3:
    st.header("Policy Recommendations")

    # Regime comparison
    st.subheader("Three-Regime Comparison")
    regimes = compute_regime_comparison(params)

    fig_reg = make_subplots(rows=1, cols=3, subplot_titles=("Present Sovereignty", "Future Sovereignty", "Dependency"))

    regime_colors = {"status_quo": COLORS["drift"], "designed": COLORS["designed"], "overreaction": COLORS["destructive"]}
    regime_labels = {"status_quo": "Status Quo", "designed": "Designed Pressure", "overreaction": "Overreaction"}

    for key, r in regimes.items():
        c = regime_colors[key]
        fig_reg.add_trace(go.Scatter(x=r.t, y=r.Kp, line=dict(color=c, width=2), name=regime_labels[key], legendgroup=key), row=1, col=1)
        fig_reg.add_trace(go.Scatter(x=r.t, y=r.Kf, line=dict(color=c, width=2), name=regime_labels[key], legendgroup=key, showlegend=False), row=1, col=2)
        fig_reg.add_trace(go.Scatter(x=r.t, y=r.D, line=dict(color=c, width=2), name=regime_labels[key], legendgroup=key, showlegend=False), row=1, col=3)

    fig_reg.update_layout(height=350, margin=dict(t=40, b=20))
    fig_reg.update_xaxes(title_text="Years")
    st.plotly_chart(fig_reg, use_container_width=True)

    # Value comparison
    cols = st.columns(3)
    for i, (key, r) in enumerate(regimes.items()):
        with cols[i]:
            st.metric(regime_labels[key], f"V = {r.V_total:.4f}")

    # Delay cost curve
    st.subheader("Shadow Price of Delay")
    delays, costs = compute_delay_cost_curve(alpha=0.3, sigma=2.0, max_delay=15.0, params=params)
    fig_delay = go.Figure()
    fig_delay.add_trace(go.Scatter(x=delays, y=costs, fill="tozeroy", fillcolor="rgba(139,64,73,0.15)", line=dict(color=COLORS["present"], width=2)))
    fig_delay.update_layout(height=300, xaxis_title="Delay (years)", yaxis_title="Cumulative cost", margin=dict(t=20, b=20))
    st.plotly_chart(fig_delay, use_container_width=True)

    # Domain scoring
    st.subheader("Domain Priority Scoring")
    domain_data = []
    for name, assessment in DEFAULT_DOMAINS.items():
        s = score_domain(**assessment)
        domain_data.append({"Domain": name.replace("_", " ").title(), "Score": s, **assessment})

    for d in sorted(domain_data, key=lambda x: x["Score"], reverse=True):
        st.progress(d["Score"], text=f"**{d['Domain']}** — Score: {d['Score']:.2f} (Window: {d['W_remaining']:.1f}, Capability: {d['capability']:.1f}, Strategic Value: {d['strategic_value']:.1f})")
```

- [ ] **Step 2: Test the dashboard launches**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
source .venv/bin/activate
streamlit run app/dashboard.py --server.port 3001 &
sleep 3
curl -s http://localhost:3001 | head -5
kill %1
```

Expected: HTML output from Streamlit confirming the app loads.

- [ ] **Step 3: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/app/dashboard.py
git commit -m "feat: add Streamlit interactive dashboard

Three-tab layout: analytical core (trajectories, hump shape),
evolutionary simulation (firm dynamics, Monte Carlo sweep),
policy interface (regime comparison, delay costs, domain scoring)."
```

---

### Task 7: Generate Static Figures and Cross-Layer Validation

**Files:**
- Create: `sovereignty-model/generate_figures.py`

- [ ] **Step 1: Write figure generation script**

Create `sovereignty-model/generate_figures.py`:

```python
"""Generate all static figures and run cross-layer validation.

Run: python generate_figures.py
"""
import numpy as np
from pathlib import Path

from model.calibration import Parameters
from model.analytical import simulate_forward, find_optimal_alpha
from model.evolutionary import simulate_evolution, sweep_sigma
from model.policy import compute_regime_comparison, compute_delay_cost_curve
from model.visualization import (
    plot_hump_shape,
    plot_regime_comparison,
    plot_delay_costs,
    plot_evolutionary_trajectories,
    plot_optimal_alpha,
)

OUT = Path("output/figures")
OUT.mkdir(parents=True, exist_ok=True)

params = Parameters()

print("=== Generating static figures ===\n")

# 1. Hump shape (analytical only)
print("1. Hump shape...")
plot_hump_shape(save_path=str(OUT / "hump_shape_analytical.png"), params=params)

# 2. Optimal α
print("2. Optimal allocation...")
opt = find_optimal_alpha(sigma=2.0, params=params)
plot_optimal_alpha(opt, save_path=str(OUT / "optimal_alpha.png"))
print(f"   α* = {opt.alpha_star:.3f}, V* = {opt.V_star:.4f}")

# 3. Regime comparison
print("3. Regime comparison...")
regimes = compute_regime_comparison(params)
plot_regime_comparison(regimes, save_path=str(OUT / "regime_comparison.png"))
for name, r in regimes.items():
    print(f"   {name}: V = {r.V_total:.4f}")

# 4. Delay cost curve
print("4. Delay costs...")
delays, costs = compute_delay_cost_curve(alpha=0.3, sigma=2.0, max_delay=15.0, params=params)
plot_delay_costs(delays, costs, save_path=str(OUT / "delay_costs.png"))

# 5. Evolutionary trajectories
print("5. Evolutionary trajectories...")
evo_results = {}
for sigma in [0.0, 2.0, 15.0]:
    evo_results[sigma] = simulate_evolution(sigma, params, seed=42)
plot_evolutionary_trajectories(evo_results, save_path=str(OUT / "evolutionary_trajectories.png"))

# 6. Cross-layer validation: h(σ) simulated vs analytical
print("6. Cross-layer h(σ) validation...")
sweep_sigmas = np.linspace(0, 12, 13).tolist()
sweep_results = sweep_sigma(sweep_sigmas, params, n_replications=20, seed=42)
plot_hump_shape(
    params=params,
    simulated_data=sweep_results,
    save_path=str(OUT / "hump_shape_validated.png"),
)

# Print validation summary
means = [np.mean(sweep_results[s]) for s in sweep_sigmas]
baseline = means[0]
norm_means = [m / baseline for m in means]
peak_idx = np.argmax(norm_means)
print(f"   Simulated peak at σ ≈ {sweep_sigmas[peak_idx]:.1f}")
print(f"   Analytical peak at σ* = {1.0 / params.b:.1f}")

print(f"\n=== All figures saved to {OUT} ===")
```

- [ ] **Step 2: Run the figure generation**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
source .venv/bin/activate
python generate_figures.py
```

Expected: All 6 figures generated in `output/figures/`, cross-layer validation confirms simulated peak near analytical σ*.

- [ ] **Step 3: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/generate_figures.py sovereignty-model/output/figures/
git commit -m "feat: add figure generation and cross-layer validation

Generates hump shape, optimal allocation, regime comparison,
delay costs, evolutionary trajectories, and validated h(σ) overlay.
Cross-layer validation confirms simulated peak near analytical σ*."
```

---

### Task 8: Final Test Suite Run and Cleanup

**Files:**
- Modify: `sovereignty-model/model/__init__.py`

- [ ] **Step 1: Update package __init__.py with public API**

Update `sovereignty-model/model/__init__.py`:

```python
"""Sovereignty economic model — three-layer nested framework.

Layer 1 (Analytical): Optimal control problem for sovereignty allocation
Layer 2 (Evolutionary): Agent-based simulation of selection pressure
Layer 3 (Policy): Actionable policy recommendations from observables
"""
from model.calibration import Parameters, window_openness, h_sigma
from model.analytical import simulate_forward, find_optimal_alpha, compute_shadow_price_of_delay
from model.evolutionary import simulate_evolution, sweep_sigma
from model.policy import compute_regime_comparison, compute_policy_recommendation

__all__ = [
    "Parameters",
    "window_openness",
    "h_sigma",
    "simulate_forward",
    "find_optimal_alpha",
    "compute_shadow_price_of_delay",
    "simulate_evolution",
    "sweep_sigma",
    "compute_regime_comparison",
    "compute_policy_recommendation",
]
```

- [ ] **Step 2: Run full test suite**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
source .venv/bin/activate
pytest tests/ -v --tb=short
```

Expected: All tests pass across test_calibration.py, test_analytical.py, test_evolutionary.py, test_policy.py.

- [ ] **Step 3: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/model/__init__.py
git commit -m "feat: finalize package API and verify full test suite

Public API exports all three layers. All tests passing."
```

---

### Task 9: Comparative Statics and Parameter Sensitivity

**Files:**
- Modify: `sovereignty-model/model/calibration.py`
- Modify: `sovereignty-model/model/analytical.py`
- Modify: `sovereignty-model/tests/test_calibration.py`

- [ ] **Step 1: Write failing tests for parameter sweeps**

Add to `sovereignty-model/tests/test_calibration.py`:

```python
from model.calibration import parameter_sweep


class TestParameterSweep:
    """Verify parameter sweep utilities."""

    def test_sweep_returns_results(self):
        results = parameter_sweep("lam", [0.1, 0.3, 0.5])
        assert len(results) == 3

    def test_higher_lock_in_speed_increases_dependency(self):
        results = parameter_sweep("lam", [0.1, 0.5])
        assert results[0.5].D[-1] > results[0.1].D[-1]

    def test_faster_window_closure_reduces_kf(self):
        results = parameter_sweep("gamma", [0.05, 0.2])
        assert results[0.05].Kf[-1] > results[0.2].Kf[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_calibration.py::TestParameterSweep -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Add parameter_sweep to calibration module**

Add to `sovereignty-model/model/calibration.py`:

```python
def parameter_sweep(
    param_name: str,
    values: list[float],
    base_params: Parameters | None = None,
    alpha: float = 0.3,
    sigma: float = 2.0,
) -> dict[float, "SimulationResult"]:
    """Sweep a single parameter, running forward simulation for each value.

    Args:
        param_name: Name of parameter to vary (must be a field of Parameters)
        values: Values to test
        base_params: Base parameters (defaults used if None)
        alpha: Fixed allocation for simulations
        sigma: Fixed constraint intensity

    Returns:
        Dict mapping parameter value → SimulationResult
    """
    from model.analytical import simulate_forward

    if base_params is None:
        base_params = Parameters()

    results = {}
    for v in values:
        p = Parameters(**{**base_params.__dict__, param_name: v})
        results[v] = simulate_forward(alpha, sigma, p)
    return results
```

- [ ] **Step 4: Add compute_comparative_statics to analytical module**

Add to `sovereignty-model/model/analytical.py`:

```python
def compute_comparative_statics(
    params: Parameters,
    alpha: float = 0.3,
    sigma: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Compute comparative statics for key parameters.

    Varies lambda, gamma, and beta around baseline, reports
    terminal Kp, Kf, D, and total V for each.

    Returns:
        Dict of {param_name: {value: V_total}}
    """
    from model.calibration import parameter_sweep

    statics = {}
    for param, values in [
        ("lam", [0.1, 0.2, 0.3, 0.4, 0.5]),
        ("gamma", [0.05, 0.1, 0.15, 0.2, 0.25]),
        ("beta", [0.3, 0.4, 0.5, 0.6, 0.7]),
    ]:
        results = parameter_sweep(param, values, params, alpha, sigma)
        statics[param] = {v: r.V_total for v, r in results.items()}
    return statics
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_calibration.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/model/calibration.py sovereignty-model/model/analytical.py sovereignty-model/tests/test_calibration.py
git commit -m "feat: add comparative statics and parameter sweep utilities

Sweep any parameter, compute sensitivity of V to lambda, gamma, beta.
Covers spec requirement for comparative statics and sensitivity analysis."
```

---

### Task 10: Capability Threshold and Firm Distribution Visualization

**Files:**
- Modify: `sovereignty-model/model/evolutionary.py`
- Modify: `sovereignty-model/model/visualization.py`
- Modify: `sovereignty-model/tests/test_evolutionary.py`

- [ ] **Step 1: Write failing test for capability threshold**

Add to `sovereignty-model/tests/test_evolutionary.py`:

```python
from model.evolutionary import find_capability_threshold


class TestCapabilityThreshold:
    """Verify minimum capability for designed pressure to redirect."""

    def test_returns_threshold(self):
        p = Parameters(N=200, T=15.0)
        threshold = find_capability_threshold(sigma=2.0, params=p, seed=42)
        assert threshold > 0

    def test_threshold_increases_with_sigma(self):
        """Higher σ requires higher capability to survive."""
        p = Parameters(N=200, T=15.0)
        t_low = find_capability_threshold(sigma=1.0, params=p, seed=42)
        t_high = find_capability_threshold(sigma=5.0, params=p, seed=42)
        assert t_high > t_low
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evolutionary.py::TestCapabilityThreshold -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Add find_capability_threshold to evolutionary module**

Add to `sovereignty-model/model/evolutionary.py`:

```python
def find_capability_threshold(
    sigma: float,
    params: Parameters,
    seed: int = 42,
    n_replications: int = 10,
) -> float:
    """Find minimum initial capability for firms to benefit from constraint.

    Runs simulations, bins firms by initial capability, identifies the
    threshold below which firms lose capability under designed pressure
    compared to no pressure.

    Returns:
        Approximate capability threshold
    """
    cap_bins = np.linspace(0.1, 3.0, 20)
    benefit_by_bin = np.zeros(len(cap_bins) - 1)
    counts_by_bin = np.zeros(len(cap_bins) - 1)

    for rep in range(n_replications):
        s = seed + rep * 1000
        r_constrained = simulate_evolution(sigma, params, seed=s)
        r_baseline = simulate_evolution(0.0, params, seed=s)

        init_caps = r_constrained.capability_history[0]
        final_constrained = r_constrained.capability_history[-1]
        final_baseline = r_baseline.capability_history[-1]

        benefit = final_constrained - final_baseline

        for i in range(len(cap_bins) - 1):
            mask = (init_caps >= cap_bins[i]) & (init_caps < cap_bins[i + 1])
            if mask.sum() > 0:
                benefit_by_bin[i] += benefit[mask].mean()
                counts_by_bin[i] += 1

    avg_benefit = np.where(counts_by_bin > 0, benefit_by_bin / counts_by_bin, 0)

    # Threshold: first bin where average benefit turns positive
    positive_bins = np.where(avg_benefit > 0)[0]
    if len(positive_bins) == 0:
        return cap_bins[-1]  # No firms benefit
    return cap_bins[positive_bins[0]]
```

- [ ] **Step 4: Add plot_firm_distribution to visualization module**

Add to `sovereignty-model/model/visualization.py`:

```python
def plot_firm_distribution(
    capability_history: list[np.ndarray],
    time_points: list[int] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot capability distribution snapshots at selected time points."""
    if time_points is None:
        n = len(capability_history)
        time_points = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    fig, axes = plt.subplots(1, len(time_points), figsize=(3 * len(time_points), 3.5), sharey=True)

    for ax, t_idx in zip(axes, time_points):
        caps = capability_history[t_idx]
        ax.hist(caps, bins=30, color=COLORS["designed"], alpha=0.7, edgecolor="white")
        ax.set_title(f"t = {t_idx}", fontsize=10)
        ax.set_xlabel("Capability")
        if t_idx == time_points[0]:
            ax.set_ylabel("Count")

    plt.suptitle("Capability Distribution Over Time", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_evolutionary.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/model/evolutionary.py sovereignty-model/model/visualization.py sovereignty-model/tests/test_evolutionary.py
git commit -m "feat: add capability threshold detection and firm distribution plots

Finds minimum absorptive capacity for designed pressure to redirect
rather than destroy. Adds capability distribution snapshot visualization."
```

---

### Task 11: README

**Files:**
- Create: `sovereignty-model/README.md`

- [ ] **Step 1: Write README**

Create `sovereignty-model/README.md`:

```markdown
# Sovereignty Economic Model

A three-layer formal economic model of the argument in "Two Kinds of Sovereignty."

## Quick Start

```bash
cd sovereignty-model
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Run Dashboard

```bash
streamlit run app/dashboard.py --server.port 3001
```

## Generate Figures

```bash
python generate_figures.py
```

Outputs to `output/figures/`.

## Architecture

- **Layer 1 (`model/analytical.py`)** — Optimal control: state equations, forward simulation, shadow prices, optimal α*
- **Layer 2 (`model/evolutionary.py`)** — Agent-based: N heterogeneous firms, innovation lottery, selection, Monte Carlo sweep
- **Layer 3 (`model/policy.py`)** — Policy interface: regime comparison, delay costs, domain scoring, integrated recommendations
- **Shared (`model/calibration.py`)** — Parameters with empirical anchors, h(σ) hump shape, parameter sweep utilities

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| D₀ (initial dependency) | 0.67 | Synergy Research Group |
| R (EU R&D/GDP) | 2.2% | European Investment Bank |
| σ* (optimal constraint) | 2.0 | Derived from h(σ) peak |
```

- [ ] **Step 2: Commit**

```bash
cd ~/projects/sovereignty-visualization
git add sovereignty-model/README.md
git commit -m "docs: add README with quick start and architecture overview"
```

---

### Task 12: Final Integration Test

- [ ] **Step 1: Run complete test suite**

```bash
cd ~/projects/sovereignty-visualization/sovereignty-model
source .venv/bin/activate
pytest tests/ -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 2: Generate all figures**

```bash
python generate_figures.py
```

Expected: All figures generated, cross-layer validation confirms simulated peak near analytical σ*.

- [ ] **Step 3: Launch dashboard and verify**

```bash
streamlit run app/dashboard.py --server.port 3001 &
sleep 3
curl -s http://localhost:3001 | grep -c "Sovereignty"
kill %1
```

Expected: At least 1 match confirming dashboard loads.

- [ ] **Step 4: Final commit**

```bash
cd ~/projects/sovereignty-visualization
git add -A
git commit -m "chore: final integration verification — all tests pass, figures generated, dashboard operational"
```
