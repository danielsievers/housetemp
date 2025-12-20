# House Temp Prediction (Home Assistant Component)

This custom component integrates the HouseTemp thermal model into Home Assistant, providing indoor temperature predictions based on your home's physics, weather forecast, and HVAC schedule.

## Installation

### HACS (Recommended)
1.  Open HACS in Home Assistant.
2.  Go to **Integrations** > Top right menu > **Custom repositories**.
3.  Enter the URL of this repository.
4.  Category: **Integration**.
5.  Click **Add** and then **Download**.
6.  Restart Home Assistant.

*Note: The core logic is now included with the component, so no external library installation is required.*

### Manual Installation
1.  Copy the `housetemp` directory from `custom_components/` to your Home Assistant's `custom_components` directory.
2.  Restart Home Assistant.

## Configuration

1.  Go to **Settings > Devices & Services**.
2.  Click **Add Integration**.
3.  Search for **House Temp Prediction**.
4.  Follow the configuration steps:
    - **Sensors**: Select your Indoor Temperature sensor and Weather entity. Optionally select a Solar Forecast sensor.
    - **Parameters**: Enter your model parameters (`C_thermal`, `UA`, etc.) derived from the CLI tool.
    - **Configs**: Paste your Heat Pump JSON and Schedule JSON configuration.
    - **Settings**: Set the forecast duration (hours) and update interval (minutes).
    - **Smart Wake-Up**: The system automatically schedules a pre-heating optimization **12 hours before your return**. This ensures the house is warm when you arrive without using inefficient emergency heating.

## Away Mode & Smart Wake-Up

The system supports a robust "Away Mode" that overrides the schedule with a safety temperature (e.g., 50°F) for extended durations.

1.  **Immediate Action**: Upon setting "Away", the system re-optimizes with the safety temperature as the target.
2.  **Smart Wake-Up**: A background timer is scheduled for **12 hours before** the return time. When this fires, the unique optimization algorithm "sees" the return setpoint and plans a gradual, efficient pre-heating ramp properly timed for your arrival.
3.  **Persistence**: The Away state (End Time, Safety Temp) and the Smart Wake-Up timer are persisted to survive system restarts.

### Energy Estimation
When `set_away` is activated, the system estimates:
*   Consumption if the original schedule were followed.
*   Consumption using the new Away Setpoint (and smart return).
*   Allows calculation of "Savings" from the away action.

## Services

### Run HVAC Optimization
`housetemp.run_hvac_optimization`
Manually triggers the optimization process for a specific duration. Useful for testing or forcing a recalculation.
- **Targeting**: Requires an `entity_id` (e.g., `sensor.indoor_temp_forecast`).

### Set Away Mode
`housetemp.set_away`
Temporarily overrides the schedule for a duration.

- **Targeting**: Requires an `entity_id`.
- **Arguments**:
  - `duration`: Time to stay in away mode (e.g., "7 days", "12 hours").
  - `safety_temp`: Temperature to maintain (e.g., 50°F).

## Configuration Reference

### Schedule JSON (`comfort.json`)
The **Schedule JSON** configuration expects a format like this. You can use this to define your standard heating/cooling schedule.

```json
{
    "center_preference": 0.5,
    "mode": "heat",
    "schedule": [
        {"time": "08:00", "temp": 70},
        {"time": "22:00", "temp": 60}
    ]
}
```

## Entities

The integration creates a sensor: `sensor.indoor_temperature_forecast`.
- **State**: The predicted temperature at the end of the forecast period.
- **Attributes**:
    - `forecast`: A list of hourly/30-min predictions including temperature, HVAC state, and estimated energy usage.
    - `away` (boolean): `true` when Away Mode is active.
    - `away_end` (datetime): The local time when Away Mode is scheduled to end.

## Energy Optimization Tips

### True-Off Setpoint Bounds

The optimizer can signal "HVAC off" by pushing setpoints to the configured minimum (heating) or maximum (cooling). To maximize savings:

1. **Set `min_setpoint` 1°F below your thermostat's minimum** (e.g., if thermostat min is 55°F, set `min_setpoint` to 54°F)
2. **Set `max_setpoint` 1°F above your thermostat's maximum** (for cooling)

When the optimizer outputs these "impossible" setpoints, use an automation to turn OFF the HVAC:

```yaml
automation:
  - alias: "HVAC Off when optimizer signals off"
    trigger:
      - platform: numeric_state
        entity_id: sensor.indoor_temperature_forecast
        below: 55  # Your thermostat's actual minimum
    action:
      - service: climate.turn_off
        target:
          entity_id: climate.thermostat
```

This eliminates idle/standby power during unoccupied periods, improving energy savings accuracy.

## Testing

To run the unit tests for this component:

1.  Set up the development environment (at project root):
    ```bash
    make setup
    ```
2.  Run tests using the virtual environment:
    ```bash
    # Use Makefile
    make test

    # Or manually with venv python
    .venv/bin/python -m pytest tests/
    ```

## Visualization

### ApexCharts Card
Here is an example configuration for the [ApexCharts Card](https://github.com/RomRider/apexcharts-card):

```yaml
type: custom:apexcharts-card
graph_span: 24h
span:
  start: hour
header:
  show: false

yaxis:
  - id: temp
    opposite: false
    decimals: 1
    min: 60
    max: 75
  - id: energy
    opposite: true
    decimals: 1
    min: 0
    max: 2.5

series:
  - entity: sensor.indoor_temperature_forecast
    name: Energy (Hourly)
    type: column
    yaxis_id: energy
    unit: kWh
    color: "#775DD0"
    extend_to: false
    data_generator: |
      const pts = (entity.attributes.energy_per_hour || []);
      return pts
        .filter(p => p && p.datetime && p.kwh !== undefined && p.kwh !== null)
        .map(p => [new Date(p.datetime).getTime(), Number(p.kwh)]);

  - entity: sensor.indoor_temperature_forecast
    name: Predicted
    type: line
    yaxis_id: temp
    stroke_width: 2
    color: orange
    extend_to: false
    data_generator: |
      return (entity.attributes.forecast || []).map(p => [
        new Date(p.datetime).getTime(),
        p.temperature
      ]);

  - entity: sensor.indoor_temperature_forecast
    name: Target
    type: line
    yaxis_id: temp
    stroke_width: 2
    color: steelblue
    curve: stepline
    extend_to: false
    data_generator: |
      return (entity.attributes.forecast || []).map(p => [
        new Date(p.datetime).getTime(),
        p.target_temp
      ]);

  - entity: sensor.indoor_temperature_forecast
    name: Optimized
    type: line
    yaxis_id: temp
    stroke_width: 2
    color: "#00E396"
    curve: stepline
    extend_to: false
    data_generator: |
      return (entity.attributes.forecast || [])
        .filter(p => p.ideal_setpoint !== undefined && p.ideal_setpoint !== null)
        .map(p => [new Date(p.datetime).getTime(), p.ideal_setpoint]);

apex_config:
  chart:
    height: 300
    stacked: false
  plotOptions:
    bar:
      columnWidth: "70%"
  markers:
    size: 0
  stroke:
    width: [0, 2, 2, 2]
  xaxis:
    type: datetime
    tooltip:
      enabled: false
    labels:
      datetimeFormatter:
        hour: "ha"
  yaxis:
    - tickAmount: 6
      decimalsInFloat: 0
      title:
        text: "Temp (°F)"
    - tickAmount: 6
      opposite: true
      decimalsInFloat: 1
      title:
        text: "Energy (kWh/hr)"
  tooltip:
    shared: true
    x:
      format: dd MMM h:mm tt
```

