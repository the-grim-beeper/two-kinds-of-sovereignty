# Design Spec: Formal Economic Model of "Two Kinds of Sovereignty"

## Overview

A three-layer formal economic model that unifies the core mechanisms of the essay "Two Kinds of Sovereignty" — the exploitation/exploration allocation tradeoff, the selection pressure typology, and Perez window dynamics — into a single integrated framework. The model produces both a formal analytical write-up (LaTeX-ready) and an interactive computational companion for policy audiences.

**Primary audience:** Policy practitioners and technically literate advisors. The model must be rigorous but optimized for actionable insight — optimal allocation ratios, threshold conditions, policy design principles — rather than theoretical novelty per se.

## Theoretical Foundations

The model draws on and integrates:

- **March (1991)** — exploitation vs. exploration as the organizational tension underlying present vs. future sovereignty
- **Arthur (1989)** — increasing returns to adoption and path dependence as the mechanism that locks in dependency
- **Hicks (1932), Hayami & Ruttan (1971)** — induced innovation hypothesis: technical change directed by relative scarcity of inputs
- **Perez (2002)** — techno-economic paradigm windows that open during installation periods and close as standards lock in
- **Acemoglu (2002)** — directed technical change framework as the analytical template
- **Nelson & Winter (1982)** — evolutionary economics and heterogeneous firm dynamics

## Layer 1: Analytical Core — The Sovereignty Allocation Problem

### Setup

A social planner allocates a unit resource flow `R` between two activities at each moment `t`:

- **Exploitation** `αR` — secures and domesticates existing technology. Produces present sovereignty capital `Kp` with diminishing returns, governed by path dependence.
- **Exploration** `(1-α)R` — invests in new technological paradigms. Produces future sovereignty capital `Kf` through a stochastic innovation process.

### State Variables

| Variable | Description | Range |
|----------|-------------|-------|
| `Kp(t)` | Present sovereignty stock (cloud resilience, domesticated infra, regulatory control) | ≥ 0 |
| `Kf(t)` | Future sovereignty stock (frontier R&D capacity, patent portfolio, nascent paradigms) | ≥ 0 |
| `D(t)` | Dependency level | [0, 1] |
| `W(t)` | Window openness for paradigm `j` | [0, 1] |

### Dynamics

**Present sovereignty accumulation:**
```
dKp/dt = A·f(α·R) - δp·Kp - φ(D)·Kp
```
- `A` — total factor productivity for exploitation activities
- `f(·)` — concave production function: `f(x) = x^κ` with `κ ∈ (0, 1)` (Cobb-Douglas, diminishing returns to exploitation)
- `δp` — base depreciation rate (external systems evolve, rendering domestic alternatives obsolete)
- `φ(D)` — dependency-accelerated depreciation: the more dependent you are, the faster domestic capacity erodes. Increasing in `D`.

**Future sovereignty accumulation (single-domain simplification):**

For analytical tractability, Layer 1 treats the economy as targeting a single paradigm window at a time. The multi-domain extension (portfolio over `j` windows) is handled in Layer 3's domain selection instrument.

```
dKf/dt = W(t)·g((1-α)·R, σ) - δf·Kf
```
- `W(t)` — window openness for the targeted paradigm. Exploration investment has zero marginal return once the window closes.
- `g(·, σ)` — exploration production function, modified by the constraint instrument
- `δf` — depreciation of future sovereignty capital (knowledge decay, talent attrition)

**Dependency dynamics (Arthur increasing returns):**
```
dD/dt = λ·(D̄ - D)·D - η·Kp·D
```
- Logistic process toward lock-in ceiling `D̄`
- `λ` — speed of lock-in (increasing returns parameter)
- `η` — rate at which present sovereignty investment slows dependency growth
- The `η·Kp·D` term (multiplicative in `D`) ensures `D` remains in `[0, 1]`: when `D → 0`, the reduction term vanishes, preventing negative dependency. Implementation clamps `D` to `[0, 1]` as a numerical safeguard.

**Window dynamics (Perez process):**
```
W_j(t) = W₀_j · exp(-γ_j · max(0, t - t_open_j))  for t > t_open_j
```
- Windows open at `t_open_j` with initial openness `W₀_j`
- Close exponentially at rate `γ_j` (domain-specific)
- Once closed (`W_j < ε`), exploration in domain `j` yields nothing

### The Constraint Instrument σ

The essay's "virtual export restriction." Enters the exploration production function multiplicatively:

```
g((1-α)·R, σ) = [(1-α)·R]^β · h(σ)
```

Where `h(σ)` is non-monotonic (the core formal claim):

- `h(0) = 1` — no constraint, baseline productivity
- `h(σ)` increasing for `σ ∈ (0, σ*)` — designed pressure raises exploration productivity (induced innovation)
- `h(σ)` decreasing for `σ > σ*` — excessive constraint becomes destructive (sanctions effect)

Functional form: `h(σ) = 1 + a·σ·exp(-b·σ)` where `a, b > 0` parameterize the hump.

This hump shape formally expresses the essay's three lanes:
- Absent pressure: `σ = 0`
- Designed pressure: `σ ≈ σ*`
- Destructive pressure: `σ >> σ*`

### Planner's Problem

```
max_{α(t), σ(t)} ∫₀^∞ e^{-ρt} · V(Kp, Kf, D) dt
```

Subject to the state equations above. The value function `V` weights present and future sovereignty with dependency as a penalty:

```
V(Kp, Kf, D) = ω·Kp + (1-ω)·Kf - ψ·D²
```

- `ω` — relative weight on present vs. future sovereignty
- `ψ` — penalty for dependency (convex, reflecting non-linear costs of high dependency)

**Rationale for linear-quadratic form:** Sovereignty benefits are linear because each unit of `Kp` or `Kf` provides roughly constant marginal strategic value (resilience, optionality). Dependency costs are quadratic because the damage from dependency is convex — going from 50% to 60% dependent is substantially worse than 20% to 30%, reflecting the non-linear lock-in dynamics and the narrowing of exit options at high dependency. This functional form also ensures the optimal policy is smooth (interior solution) rather than bang-bang, which is more realistic for gradual policy reallocation.

### Key Analytical Outputs

1. **Optimal allocation path `α*(t)`** — when should the planner shift from exploitation-heavy to exploration-heavy?
2. **Optimal constraint intensity `σ*(t)`** — what level of designed pressure, and how does it evolve?
3. **Shadow price of delay** — the cost of waiting one period to begin exploration, as a function of window state
4. **Threshold conditions** — the dependency level `D` and window state `W` at which the planner should switch regimes
5. **Comparative statics** — how do outcomes respond to changes in `λ` (lock-in speed), `γ` (window closure rate), and `β` (exploration returns)?

## Layer 2: Evolutionary Simulation — Selection Pressure Dynamics

### Purpose

Relaxes the representative agent and smooth production function assumptions. Populates the economy with heterogeneous firms to capture emergent dynamics: why some constraint regimes produce compounding capability while others fragment the innovation system.

### Agents

`N` firms (default `N = 500`), each characterized by:

| Attribute | Description | Initial distribution |
|-----------|-------------|---------------------|
| `cᵢ(t)` | Capability stock (absorptive capacity, tacit knowledge) | Log-normal |
| `rᵢ(t)` | Research orientation [0,1] from exploitation to exploration | Beta(2, 5) — skewed toward exploitation |
| `xᵢ(t)` | External dependency ratio | Beta(5, 2) — skewed toward high dependency |

### Innovation Process

Each period (period length `Δt = 1` year, mapping to Layer 1's continuous time via Euler discretization), firm `i` draws from an innovation lottery:

**Exploitation draws:**
- Probability of success: `p_exploit = p₀ · cᵢ^ν`
- Payoff: `Δcᵢ = ε_exploit · cᵢ · xᵢ` (leveraging external inputs)
- Character: high probability, low variance. Incremental improvement.

**Exploration draws:**
- Probability of success: `p_explore = q₀ · cᵢ^ν · W(t)`
- Payoff: `Δcᵢ = ε_explore · cᵢ · (1 - xᵢ)` (building on internal capacity)
- Character: low probability, high variance. Novel capability that reduces dependency.
- `ε_explore` drawn from a fat-tailed distribution (Pareto with shape `ξ = 2.5`) — rare breakthroughs.

**Layer 2 innovation parameters:**

| Parameter | Description | Default | Calibration rationale |
|-----------|-------------|---------|----------------------|
| `p₀` | Base exploitation success probability | 0.7 | Most exploitation attempts yield incremental gains |
| `q₀` | Base exploration success probability | 0.15 | Exploration is inherently uncertain |
| `ν` | Capability exponent in success probability | 0.3 | Moderate absorptive capacity effect (diminishing) |
| `ε_exploit` | Exploitation payoff scale | 0.05 | Small incremental gains per success |
| `ε_explore` | Exploration payoff location (Pareto) | 0.1 | Base exploration payoff; tail generates breakthroughs |
| `ξ` | Exploration payoff shape (Pareto) | 2.5 | Fat-tailed but finite variance |

**Mapping to Layer 1:** The aggregate innovation production function emerging from Layer 2 should approximate Layer 1's `g(·)`. Specifically: `β` from Layer 1 corresponds to the aggregate elasticity of exploration output with respect to exploration input, which emerges from `ν`, `q₀`, and the capability distribution. The calibration module verifies this correspondence numerically.

**Constraint regime `σ` modifies draws:**
- Firms with high `xᵢ` face an immediate productivity shock: exploitation returns scaled by `(1 - σ·xᵢ)`
- Firms with sufficient `cᵢ` redirect toward exploration (induced innovation at firm level)
- Firms with low `cᵢ` cannot redirect — they substitute with inferior alternatives

### Selection Mechanism

Firms compete for market share `sᵢ` based on capability-weighted output:

```
s̃ᵢ(t+1) = sᵢ(t) · (cᵢ(t) / c̄(t))^θ
sᵢ(t+1) = s̃ᵢ(t+1) / Σⱼ s̃ⱼ(t+1)    [renormalize to sum to 1]
```

- `θ` — selection intensity (default `θ = 0.5`)
- Market shares are renormalized each period to maintain `Σᵢ sᵢ = 1`
- Firms with `sᵢ < s_min` exit; new entrants replace them with drawn-from-distribution capabilities
- Aggregate capability: `C(t) = Σᵢ sᵢ(t) · cᵢ(t)`

### Emergent Regimes

Under different `σ` values, three qualitatively distinct system dynamics emerge:

1. **σ = 0 (absent pressure):** Firms drift toward high dependency. Aggregate capability grows slowly, undirected. Efficient but fragile.

2. **σ moderate (designed pressure):** High-capability firms redirect toward exploration. Some low-capability firms fail, but survivors develop alternative pathways. Aggregate capability dips then compounds. Clustering around new paradigms emerges.

3. **σ excessive (destructive pressure):** Most firms lack absorptive capacity to redirect. Mass substitution with inferior alternatives. Aggregate capability degrades. Knowledge spillover networks sever.

### Calibration Link to Layer 1

- Takes shared calibration parameters (`δp`, `δf`, `λ`, `D̄`, window parameters) as inputs. Layer 1's `β` is not directly injected but rather validated: the emergent aggregate exploration elasticity from the simulation should approximate `β`.
- Validates `h(σ)` hump shape computationally: sweep `σ`, measure aggregate capability at terminal time `T`
- Reveals what the analytical model cannot:
  - Variance of outcomes under each regime
  - Minimum absorptive capacity threshold below which designed pressure degrades
  - Role of firm heterogeneity (Gini of capability distribution) in system resilience

### Key Simulation Outputs

1. **h(σ) validation** — empirical hump shape from Monte Carlo sweeps
2. **Capability threshold** — minimum `cᵢ` for designed pressure to redirect rather than destroy
3. **Optimal constraint graduation** — should `σ` ramp gradually or apply as a step function?
4. **Heterogeneity sensitivity** — how does the capability distribution shape affect system response?
5. **Transition dynamics** — the dip-then-compound pattern under designed pressure, with confidence intervals

## Layer 3: Policy Design Interface — The Strategic Necessity Instrument

### Purpose

Translates formal results from Layers 1 and 2 into an actionable decision framework. Given observable state variables, what should the planner do?

### Observable State Variables

The planner doesn't observe `Kf` or `W(t)` directly. Observable proxies:

| Observable | Proxy for | Empirical anchor |
|-----------|-----------|-----------------|
| `D_obs` — dependency indicators | `D(t)` | Cloud market share (65-70%), import ratios for critical inputs |
| `C_obs` — capability indicators | Firm capability distribution | R&D intensity (2.2% GDP), STEM workforce, absorptive capacity |
| `W_obs` — window signals | `W(t)` | Technology maturity curves, standards-setting activity, private VC acceleration |

### Policy Instruments

**Instrument 1 — Allocation ratio `α*`:**
How much sovereignty spending goes to present vs. future. The analytical model provides the optimal path; the interface provides a lookup table: given current `(D_obs, C_obs, W_obs)`, output `α*` and its sensitivity to measurement error in each observable.

**Instrument 2 — Constraint intensity `σ*`:**
The virtual export restriction strength. Characterized as a function of the capability distribution — the same `σ` that redirects a high-capacity system destroys a low-capacity one. The interface provides:
- Recommended `σ*` given `C_obs`
- Safety margin (how far below the destructive threshold)
- Graduation schedule (ramp rate)

**Instrument 3 — Domain selection:**
Which paradigm windows to target. Scores candidate domains on:
- `W_j` — estimated window remaining
- `C_j` — existing European capability in domain `j`
- `S_j` — strategic value if window is captured

Optimal domain portfolio maximizes expected future sovereignty subject to a budget constraint.

### Key Policy Outputs

1. **Delay cost curves** — for each domain, the shadow price of waiting one more year. Non-linear because windows close. The formal Perez argument.

2. **Constraint design principles** — conditions under which `σ > 0` dominates `σ = 0`:
   - Sufficient absorptive capacity (above threshold from Layer 2)
   - Window still open (marginal return to exploration > 0)
   - Dependency above a critical level (low dependency makes constraint unnecessary)

3. **Regime comparison dashboards** — trajectories under:
   - Status quo: `σ = 0`, `α` high (presentism)
   - Designed pressure: `σ = σ*`, `α` rebalanced (strategic necessity)
   - Overreaction: `σ` excessive (protectionism)

## Computational Architecture

### Stack

Python throughout. NumPy/SciPy for computation, Matplotlib for static figures, Plotly for interactive, Streamlit for the dashboard.

### Module Structure

```
sovereignty-model/
├── model/
│   ├── __init__.py
│   ├── analytical.py      # Layer 1: HJB equation, optimal control
│   ├── evolutionary.py    # Layer 2: Agent-based simulation
│   ├── policy.py          # Layer 3: Policy lookup and scoring
│   ├── calibration.py     # Shared parameters, empirical anchors
│   └── visualization.py   # Plotting for all layers
├── app/
│   └── dashboard.py       # Streamlit interactive companion
├── tests/
│   ├── test_analytical.py
│   ├── test_evolutionary.py
│   ├── test_policy.py
│   └── test_calibration.py
├── output/
│   ├── figures/           # Static figures for paper
│   └── latex/             # LaTeX-ready formal write-up
└── README.md
```

### Module Responsibilities

**`model/analytical.py`**
- Implements HJB equation numerically via finite difference method
- Solves for optimal `α*(t)`, `σ*(t)` paths
- Computes shadow prices and comparative statics
- Special-case closed-form solutions where tractable (e.g., zero dependency, fully open window)
- Uses SciPy `solve_ivp` for state equation integration, `minimize` for static optimization

**`model/evolutionary.py`**
- Firm class with capability, orientation, dependency attributes
- Simulation loop: innovation draws → selection → entry/exit → aggregate
- Monte Carlo sweep function: run `M` replications across `σ` grid
- Vectorized with NumPy (firms as structured arrays, not individual objects)
- Exports: capability trajectories, distributional statistics, regime classification

**`model/policy.py`**
- Takes outputs from analytical and evolutionary layers
- Constructs policy lookup tables: `(D_obs, C_obs, W_obs) → (α*, σ*, domain_scores)`
- Computes delay cost curves per domain
- Regime comparison: generates trajectory triples (status quo, designed, overreaction)

**`model/calibration.py`**
- Central parameter definitions with documented sources
- Empirical anchors:
  - `D₀ = 0.67` (cloud dependency, Synergy Research Group)
  - `R_EU = 0.022` (R&D/GDP, EIB)
  - `R_US = 0.035`, `R_KR = 0.049` (comparative)
  - Patent trajectory slopes from EPO data
- Parameter sweep utilities for sensitivity analysis
- Mapping functions between Layer 1 and Layer 2 parameterizations

**`model/visualization.py`**
- `plot_optimal_paths()` — α*(t), σ*(t) trajectories
- `plot_hump_shape()` — h(σ) from both analytical and simulated
- `plot_capability_trajectories()` — aggregate capability under three regimes
- `plot_delay_costs()` — shadow price curves per domain
- `plot_firm_distribution()` — capability distribution snapshots
- Dual output: Matplotlib (paper) and Plotly (dashboard)

**`app/dashboard.py`**
- Three-panel Streamlit layout corresponding to three layers
- Parameter sliders: `α`, `σ`, initial `D`, capability distribution shape, window parameters
- Real-time trajectory updates on parameter change
- Scenario comparison: save and overlay parameter configurations
- Export: downloadable figures and parameter sets

### Testing Strategy

- **Analytical:** Verify against known special cases (zero dependency → trivial solution; fully open window → standard Ramsey; no constraint → baseline growth)
- **Evolutionary:** Convergence tests (results stable as `N` and `M` increase); regime classification accuracy
- **Policy:** Verify lookup tables produce sensible recommendations at boundary cases (zero dependency, fully closed windows, maximum capability). Test delay cost monotonicity (waiting always costs more as window narrows).
- **Calibration:** Parameter sensitivity — which parameters most affect policy recommendations?
- **Cross-layer:** Confirm `h(σ)` shape matches between analytical assumption and simulation output

### Empirical Calibration Anchors

| Parameter | Value | Source |
|-----------|-------|--------|
| Initial dependency `D₀` | 0.67 | Synergy Research Group (cloud market share) |
| EU R&D intensity | 0.022 | European Investment Bank |
| US R&D intensity | 0.035 | OECD |
| South Korea R&D intensity | 0.049 | OECD |
| EU AI patent trend | Declining slope 2018-2023 | EPO |
| Quantum value at stake (2030s) | $450-850B | McKinsey Global Institute 2024 |
| Window domains | Scientific AI, Quantum, Fusion | Essay argument + literature |

## Deliverables

1. **Formal write-up** — LaTeX-ready markdown with model specification, propositions (with proofs where tractable, numerical verification otherwise), calibration, and key results. Structured as a paper appendix.

2. **Computational implementation** — Python package with all three layers, reproducible results, documented parameters.

3. **Interactive dashboard** — Streamlit app deployable alongside the essay. Three panels, parameter exploration, scenario comparison. Policy audience can ask "what if we shifted X% toward structured necessity?" and see trajectory implications.

4. **Static figures** — Publication-quality figures for all key results (optimal paths, hump shape, regime comparisons, delay costs, capability distributions).
