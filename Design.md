# Design Document: HouseTemp

## 1. Goal
The goal of this system is to create a **physics-based thermal model** of a specific house using historical data (indoor temp, outdoor temp, solar radiation, HVAC usage). 

By fitting this model to reality, we extract key physical parameters (Insulation $UA$, Thermal Mass $C$) which allow us to:
1.  **Predict** future indoor temperatures based on weather forecasts.
2.  **Optimize** HVAC schedules for cost or comfort.
3.  **Estimate** energy consumption and efficiency (COP) retrospectively.

## 2. Core Physics Model
The model treats the house as a single thermal mass with heat flows in and out. The state of the system is the **Indoor Temperature** ($T_{in}$).

The fundamental differential equation governing the temperature change is:

$$ C \frac{dT_{in}}{dt} = Q_{leak} + Q_{solar} + Q_{internal} + Q_{hvac} $$

Where:
*   $C$: **Thermal Mass** (BTU/°F). Represents the house's ability to store heat.
*   $Q_{leak}$: Heat loss/gain through walls/windows.
*   $Q_{solar}$: Heat gain from the sun.
*   $Q_{internal}$: Constant heat from appliances/people.
*   $Q_{hvac}$: Active heating or cooling from the heat pump.

### 2.1 Component Equations

#### Leakage ($Q_{leak}$)
Modeled as linear conduction proportional to the temperature difference:

$$ Q_{leak} = UA \cdot (T_{out} - T_{in}) $$

*   $UA$: **Insulation Factor** (BTU/hr/°F). Lower is better insulation.

#### Solar Gain ($Q_{solar}$)
Proportional to measured solar radiation:

$$ Q_{solar} = K_{solar} \cdot S_{radiation} $$

*   $K_{solar}$: **Solar Gain Factor** (BTU/hr per kW/m²). Represents effective window area and SHGC.
*   $S_{radiation}$: Solar irradiance (kW/m²) from weather data.

#### Internal Heat ($Q_{internal}$)
Modeled as a constant background heat load:

$$ Q_{internal} = \text{Constant} \quad (\text{BTU/hr}) $$

## 3. HVAC Control Logic
The model does *not* assume the HVAC system is a simple On/Off switch. Instead, it models a **Variable Speed Inverter** heat pump.

The heat output ($Q_{hvac}$) is modeled as a Proportional Controller based on the "Gap" between the Setpoint ($T_{set}$) and Current Temp ($T_{in}$):

$$ Q_{request} = Q_{base} + (H_{factor} \cdot |T_{set} - T_{in}|) $$

*   $Q_{base}$: Minimum running capacity (e.g., 3000 BTU/hr).
*   $H_{factor}$: **Aggressiveness** (BTU/hr per °F error). How hard the unit ramps up to meet demand.

### 3.1 Constraints
The requested output is limited by two real-world factors:
1.  **Hardware Capacity**: The unit cannot exceed its maximum output at a given outdoor temperature.
2.  **Modulation Floor**: The unit defines a minimum continuous output (e.g., 12,000 BTU/hr). Any demand below this floor forces the system to **Cycle** (Pulse On/Off) rather than modulate efficiently.
3.  **System Efficiency (Derate)**: Real-world losses (e.g., duct leakage, heat exchanger scaling) reduce the effective output.

$$ Q_{hvac} = \eta_{eff} \cdot \min(Q_{request}, \text{MaxCapacity}(T_{out})) $$

*   $\eta_{eff}$: **Efficiency/Derate Factor** (0.0 - 1.0). Accounts for duct leaks and system degradation.

## 4. Parameter Optimization
We find the unknown parameters ($C, UA, K_{solar}, Q_{internal}, H_{factor}, \eta_{eff}$) by minimizing the error between the *simulated* temperature and the *actual* historical temperature.

*   **Algorithm**: L-BFGS-B (Scipy Optimize)
*   **Loss Function**: Root Mean Square Error (RMSE) of $T_{in}$.
*   **Constraints**: Physical bounds (e.g., Mass > 0, UA > 0) are enforced to prevent non-physical solutions.

## 5. Advanced Energy Estimation
Once the thermal load ($Q_{hvac}$) is known, we calculate Total Energy Consumption ($E_{total}$) using a component-based power model. The system calculates energy at each simulation time-step, allowing for granular hourly reporting and detailed efficiency analysis.

### 5.1 Power Components
The total power consumption is the sum of four distinct components:

$$ P_{total} = P_{compressor} + P_{blower} + P_{idle} + P_{defrost} $$

#### 1. Compressor Power ($P_{compressor}$)
Calculated from the thermal output ($Q_{hvac}$) and the system's efficiency:

$$ P_{compressor} = \frac{Q_{hvac} / \eta_{derate}}{\text{COP}_{effective} \cdot 3412} \quad (kW) $$

*   $\eta_{derate}$: **System Derate** (e.g., 0.75). Accounts for duct thermal losses.
*   $\text{COP}_{effective}$: Base COP $\times$ Part-Load Factor (PLF).
    *   **PLF Clipping**: PLF is clamped to $[P_{min}, 1.4]$ to prevent efficiency hallucinations at near-zero loads.

#### 2. Blower Power ($P_{blower}$)
Modeled as a discrete state based on system activity:
*   **Active**: High-static fan power (e.g., 900W) whenever $Q_{hvac} > 0$.
    *   This is an **Extra Electrical Draw** added *on top* of the Compressor input derived from COP.
    *   *Note:* Only use this if your manufacturer COP table excludes fan power (or if measuring fan separately).
    *   Default: 0.0 (Assumes COP includes fan).

#### 3. Idle Power ($P_{idle}$)
Constant sampling load (e.g., 250W) when the system is enabled but the compressor is off ($Q_{hvac} = 0$).
*   This is additive to any standby power implicit in SEER/HSPF ratings.
*   Default: 0.0.

#### 4. Defrost Penalty ($P_{defrost}$) for Heating
Modeled as a reverse-cycle penalty when $T_{out}$ falls within the risk zone (e.g., 28-42°F):
*   Logic: Applies a heavy load (e.g., 4.5 kW) for a duty cycle fraction (e.g., 10 mins/hour).

### 5.2 Thermal Inertia (Soft Start)
To prevent the optimizer from suggesting unrealistic "micro-bursts" of heat, the model imposes a **Soft Start Ramp**:
*   The first 5 minutes of any active cycle are "dampened" (linear ramp 0-100%).
*   This incentivizes longer run times by making short cycles thermally ineffective.

**Crucial Distinction:**
The model separates **Delivered Heat** (Ramped, used for $T_{in}$ simulation) from **Produced Heat** (Unramped, used for Energy Billing).
*   The compressor draws full power immediately (Produced).
*   The house receives heat slowly (Delivered).
*   This ensures the energy cost of "warming up the ducts" is correctly billed, preventing an efficiency hallucination during short cycles.

## 6. HVAC Schedule Optimization
The system can optimize the thermostat schedule to minimize energy consumption while maintaining comfort.

### 6.1 Objective Function
Minimize total cost $J$:

$$ J = \sum_{t=0}^{T} \left( \text{Cost}_{kWh}(t) + \text{Cost}_{comfort}(t) \right) $$

### 6.2 Asymmetric Comfort Penalty
Defined by a target schedule and the HVAC mode (Heat/Cool).
Let $e(t) = T_{in}(t) - T_{target}(t)$.

The cost function is asymmetric, penalizing only "bad" deviations while allowing "beneficial" overshooting (which allows the system to bank thermal energy when efficient).

*   **Heating Mode**:
    *   Goal: $T_{in} \ge T_{target}$
    *   If $T_{in} < T_{target}$: Penalty $\propto (T_{in} - T_{target})^2$
    *   If $T_{in} \ge T_{target}$: **Zero Penalty** (Overshoot is allowed).

*   **Cooling Mode**:
    *   Goal: $T_{in} \le T_{target}$
    *   If $T_{in} > T_{target}$: Penalty $\propto (T_{in} - T_{target})^2$
    *   If $T_{in} \le T_{target}$: **Zero Penalty** (Undershoot/Overcooling is allowed).

$$ \text{Cost}_{comfort} = P_{center} \cdot \left(\text{EffectiveError}\right)^2 $$

### 6.3 Precision-Guided Economy (Deadband Mode)
For installations requiring less rigid control, the system supports a **Deadband Comfort Mode**. This introduces a "Slack" region ($\pm \epsilon$) around the target where the comfort penalty is zero.

Let $T_{in}$ be current temp and $T_{set}$ be target.
*   If $|T_{in} - T_{set}| \le \text{Slack}$: $\text{Cost}_{comfort} = 0$.
*   If $|T_{in} - T_{set}| > \text{Slack}$: $\text{Cost}_{comfort} \propto (\text{Error} - \text{Slack})^2$.

This allows the HVAC to remain idle during minor thermal drifts, significantly reducing cycling and energy use without impacting perceived comfort.


## 7. Optimization Strategy
To optimize the schedule efficiently on resource-constrained hardware (e.g., Home Assistant Green / ARM), we employ a **Multi-Scale Optimization Strategy** with specific solver tuning.

### 7.1 Multi-Scale Approach (Coarse-to-Fine)
The optimization problem is non-convex and noisy. A single high-resolution pass often gets stuck in local minima or takes too long.
1.  **Pass 1 (Coarse)**: Optimizes 2-hour blocking intervals.
    *   **Goal**: Find the global "shape" of the schedule (e.g., "Pre-heat at 4 PM").
    *   **Settings**: Lazy (`eps=1.0`, `maxiter=50`). Speed is prioritized.
2.  **Pass 2 (Fine)**: Optimizes 30-minute blocking intervals.
    *   **Goal**: Refine edges and efficiency.
    *   **Initialization**: Warm-starts from the interpolated result of Pass 1.
    *   **Settings**: Precision (`eps=0.5`, `maxiter=500`).

### 7.2 Numerical Tuning
Standard solver defaults (e.g., `eps=1e-8`) fail for HVAC control because the cost function has "flat" regions (Deadbands) where small changes yield zero cost gradient.
*   **Epsilon `eps=0.5`**: Forces the solver to take "Macro Steps" (0.5°F) to "see" past the deadband walls.
*   **Tolerance `ftol=1e-4`**: Terminates early when cost improvements become negligible (< 0.01%), saving significant CPU cycles.
*   **Robustness**: The solver explicitly handles `ABNORMAL_TERMINATION_IN_LNSRCH` as a valid "Good Enough" result, preventing infinite loops in flat cost landscapes.

## 8. Data Integrity & Safety
The system is designed to be "Fail-Neutral" when encountering missing data or configuration gaps.

### 8.1 Gap Neutrality (NaN Safety)
If a simulation period lacks both an optimized plan and a fallback schedule (e.g., during configuration transitions), the system enforces a **Neutral State**:
1.  **Setpoint Clamping**: The setpoint is clamped to the current measured indoor temperature ($T_{in}[0]$).
2.  **HVAC Lockout**: Active heating/cooling is explicitly disabled (`hvac_state = 0`).

This prevents "NaN Propagation" into the physics model and ensures the forecast reflects a safely idle state rather than undefined behavior.

### 8.2 Configuration-Aware Caching
Caches of optimized schedules are keyed by a **Global Configuration Hash**. Any change to the heat pump specs, thermal parameters, or comfort settings immediately invalidates existing plans to ensure the forecast is always physically consistent with current settings.

