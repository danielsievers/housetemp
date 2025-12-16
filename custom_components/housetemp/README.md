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
  start: minute
header:
  show: true
  title: Temperature Forecast
  show_states: true
  colorize_states: true
series:
  - entity: sensor.indoor_temperature_forecast
    attribute: energy_kwh
    name: Target Energy
    unit: kWh
    color: steelblue
    float_precision: 2
    show:
      in_header: true
      in_chart: false
  - entity: sensor.indoor_temperature_forecast
    attribute: optimized_energy_kwh
    name: Opt. Energy
    unit: kWh
    color: "#00E396"
    float_precision: 2
    show:
      in_header: true
      in_chart: false
  - entity: sensor.indoor_temperature_forecast
    attribute: savings_kwh
    name: Savings
    unit: kWh
    color: green
    float_precision: 2
    show:
      in_header: true
      in_chart: false
  - entity: sensor.indoor_temperature_forecast
    name: Predicted
    type: line
    stroke_width: 2
    color: orange
    extend_to: false
    show:
      in_header: false
      in_chart: true
    data_generator: |
      return entity.attributes.forecast.map(p => [
        new Date(p.datetime).getTime(), 
        p.temperature
      ]);
  - entity: sensor.indoor_temperature_forecast
    name: Target
    type: line
    stroke_width: 2
    color: steelblue
    curve: stepline
    extend_to: false
    show:
      in_header: false
      in_chart: true
    data_generator: |
      return entity.attributes.forecast
        .map(p => [
          new Date(p.datetime).getTime(), 
          p.target_temp
        ]);
  - entity: sensor.indoor_temperature_forecast
    name: Optimized
    type: line
    stroke_width: 2
    color: "#00E396"
    curve: stepline
    extend_to: false
    show:
      in_header: false
      in_chart: true
    data_generator: |
      return entity.attributes.forecast
        .filter(p => p.ideal_setpoint !== undefined && p.ideal_setpoint !== null)
        .map(p => [
          new Date(p.datetime).getTime(), 
          p.ideal_setpoint
        ]);
apex_config:
  chart:
    height: 300
  stroke:
    width:
      - 2
      - 2
      - 2
    dashArray:
      - 0
      - 5
      - 0
  markers:
    size: 0
  xaxis:
    type: datetime
    tooltip:
      enabled: false
    labels:
      formatter: |
        EVAL: (value) => {
          return new Date(value).toLocaleTimeString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            hour: 'numeric', 
            minute: '2-digit' 
          });
        }
  tooltip:
    x:
      format: dd MMM h:mm tt
```

