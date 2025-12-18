# HouseTemp: Home Thermal Model

A physics-based thermal model for estimating home insulation (UA), thermal mass (C), and other parameters from Home Assistant history data.


## Home Assistant Integration
This repository includes a custom component for Home Assistant.
See [custom_components/housetemp/README.md](custom_components/housetemp/README.md) for installation and usage instructions.

## Usage

The main entry point is `main.py`.

### 1. Train Model (Optimization)
Fit the model parameters to your historical data.

```bash
# Basic usage (uses hardcoded defaults for initialization)
python3 main.py data.csv -o my_house.json

# Initialize with Linear Regression (recommended)
python3 main.py data.csv -r -o my_house.json
```

### 2. Prediction / Forecast
Run the model on new data (or forecast data) using saved parameters.

```bash
# Run prediction on a file
python3 main.py forecast.csv -p my_house.json

# Run prediction with a specific duration (e.g., 60 minutes)
python3 main.py forecast.csv -p my_house.json --duration 60

# Run prediction with a manual start temperature (required for forecast data without indoor_temp)
python3 main.py forecast.csv -p my_house.json --start-temp 68.0
```

### 3. Rolling Evaluation
Assess model performance by running repeated 12-hour forecasts starting at every hour in the dataset.

```bash
python3 main.py data.csv -e my_house.json
```

### 4. HVAC Schedule Optimization
Optimize your thermostat schedule to minimize energy cost while maintaining comfort.

```bash
# Optimize schedule using comfort.json
python3 main.py data.csv -p my_house.json --optimize-hvac --comfort data/comfort.json

# Optimize for a specific duration (e.g., next 24 hours)
python3 main.py data.csv -p my_house.json --optimize-hvac --comfort data/comfort.json --duration 1440
```

**Configuration (`comfort.json`):**
```json
{
    "center_preference": 0.5,
    "mode": "heat",
    "comfort_mode": "deadband",
    "deadband_slack": 1.5,
    "schedule": [
        {"time": "08:00", "temp": 70},
        {"time": "22:00", "temp": 60}
    ]
}
```



### 5. Other Arguments
- `--heat-pump <json_file>`: Path to Heat Pump configuration file (default: `data/heat_pump.json`). Required for prediction and optimization.
- `--debug-output <json_file>`: Export detailed debug results to a JSON file (useful for inspection/automation).

## Development Setup

Requires **Python 3.13+** (Home Assistant 2025.11+ requirement).

```bash
# Quick setup with make
make setup
source .venv/bin/activate

# Run tests
make test
```

Or manually:
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -v
```

> [!IMPORTANT]
> **Agents & Automation**: Always use the python executable inside the virtual environment (`.venv/bin/python`) when running commands directly, or use `make` targets. System python may not have the correct dependencies or version.
> Example: `.venv/bin/python -m pytest tests/` matches the `make test` behavior.

### Controlling Plots
By default, tests run silently without showing plots. To enable plots during tests:

```bash
SHOW_PLOTS=1 pytest tests/
```
