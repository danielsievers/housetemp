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
1.  Copy the `housetemp` directory from `homeassistant/custom_components/` to your Home Assistant's `custom_components` directory.
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

## Entities

The integration creates a sensor: `sensor.indoor_temperature_forecast`.
- **State**: The predicted temperature at the end of the forecast period.
- **Attributes**:
    - `forecast`: A list of hourly/30-min predictions including temperature, HVAC state, and estimated energy usage.

## Testing

To run the unit tests for this component:

1.  Install dependencies:
    ```bash
    pip install pytest pytest-homeassistant-custom-component
    ```
2.  Run tests:
    ```bash
    pytest tests/
    ```
