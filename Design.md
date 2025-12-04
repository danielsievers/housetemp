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
The requested output is clamped by the hardware's physical limits, which vary with Outdoor Temperature ($T_{out}$):
$$ Q_{hvac} = \min(Q_{request}, \text{MaxCapacity}(T_{out})) $$

## 4. Parameter Optimization
We find the unknown parameters ($C, UA, K_{solar}, Q_{internal}, H_{factor}$) by minimizing the error between the *simulated* temperature and the *actual* historical temperature.

*   **Algorithm**: L-BFGS-B (Scipy Optimize)
*   **Loss Function**: Root Mean Square Error (RMSE) of $T_{in}$.
*   **Constraints**: Physical bounds (e.g., Mass > 0, UA > 0) are enforced to prevent non-physical solutions.

## 5. Energy Estimation
Once the thermal load ($Q_{hvac}$) is known, we calculate Energy Consumption ($E$) using the Coefficient of Performance (COP).

$$ E_{kWh} = \frac{Q_{hvac}}{\text{COP} \cdot 3412} \cdot \Delta t_{hours} $$

*   $3412$: **Conversion Factor** (BTU per kWh).
*   The **COP** is dynamic and depends on two factors:
1.  **Outdoor Temperature**: Lower $T_{out}$ reduces efficiency (lookup table).
2.  **Part-Load Ratio**: Running at partial capacity is *more* efficient than full capacity.
    *   Modeled as a linear correction factor (e.g., +40% efficiency at 30% load).
