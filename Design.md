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
2.  **System Efficiency (Derate)**: Real-world losses (e.g., duct leakage, heat exchanger scaling) reduce the effective output.

$$ Q_{hvac} = \eta_{eff} \cdot \min(Q_{request}, \text{MaxCapacity}(T_{out})) $$

*   $\eta_{eff}$: **Efficiency/Derate Factor** (0.0 - 1.0). Accounts for duct leaks and system degradation.

## 4. Parameter Optimization
We find the unknown parameters ($C, UA, K_{solar}, Q_{internal}, H_{factor}, \eta_{eff}$) by minimizing the error between the *simulated* temperature and the *actual* historical temperature.

*   **Algorithm**: L-BFGS-B (Scipy Optimize)
*   **Loss Function**: Root Mean Square Error (RMSE) of $T_{in}$.
*   **Constraints**: Physical bounds (e.g., Mass > 0, UA > 0) are enforced to prevent non-physical solutions.

## 5. Energy Estimation
Once the thermal load ($Q_{hvac}$) is known, we calculate Energy Consumption ($E$) using the Coefficient of Performance (COP).

$$ E_{kWh} = \frac{Q_{hvac}}{\text{COP} \cdot 3412} \cdot \Delta t_{hours} $$

*   $Q_{hvac}$: **Thermal Output** (BTU). Calculated by the simulation loop and passed to the energy estimator.
*   $3412$: **Conversion Factor** (BTU per kWh).
*   The **COP** is dynamic and depends on two factors:
1.  **Outdoor Temperature**: Lower $T_{out}$ reduces efficiency (lookup table).
2.  **Part-Load Ratio**: Running at partial capacity is *more* efficient than full capacity.
    *   Modeled as a linear correction factor (e.g., +40% efficiency at 30% load).

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

## 7. Away Mode & Smart Wake-Up
The system supports a robust "Away Mode" that overrides the schedule with a safety temperature (e.g., 50°F) for extended durations.

### 7.1 Smart Wake-Up Scheduling
To prevent inefficient reheating (emergency heat / maximum power) when returning from a deep setback:
1.  **Immediate Action**: Upon setting "Away", the system re-optimizes with the safety temperature as the target.
2.  **Delayed Trigger**: A background timer is scheduled for **12 hours before** the return time (`AwayEnd`).
3.  **Pre-Heating**: When the timer fires, the optimization runs again. Since `AwayEnd` is now within the forecast horizon, the optimizer "sees" the scheduled setpoint returning.
4.  **Efficient Ramp**: The optimizer plans a gradual pre-heating ramp using the heat pump's most efficient capacity, ensuring the home is comfortable exactly upon return without energy waste.

### 7.2 Persistence
The Away state (End Time, Safety Temp) and the Smart Wake-Up timer are persisted in the configuration options to survive Home Assistant restarts.

### 7.3 Energy Estimation
When `set_away` is called, the service returns predicted energy usage if the away period is within the optimization horizon:
*   `energy_used_schedule_kwh`: Est. consumption if the original schedule were followed.
*   `energy_used_optimized_kwh`: Est. consumption using the new Away Setpoint (and smart return).
*   Allows users/automations to immediately calculate "Savings" from the away action.

### 7.4 Sensor Attributes
The main temperature sensor reflects the Away status:
*   `away` (boolean): `true` when Away Mode is active.
*   `away_end` (datetime): The local time when Away Mode is scheduled to end (only present when active).

## 8. Services & API
The integration exposes custom services to interact with the model and scheduler.

### 8.1 Safety & Targeting
All services **require** a target entity (`entity_id`). This ensures that commands are sent only to the specific HouseTemp instance intended, preventing accidental overrides in multi-zone setups. Broadcasting to all instances is not supported.

### 8.2 Available Services
*   `housetemp.run_hvac_optimization`: Manually triggers the optimization process for a specific duration.
*   `housetemp.set_away`: activating "Away Mode" with a specific duration and safety temperature.
