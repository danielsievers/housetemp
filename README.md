# HouseTemp: Home Thermal Model

A physics-based thermal model for estimating home insulation (UA), thermal mass (C), and other parameters from Home Assistant history data.
A physics-based thermal model for estimating home insulation (UA), thermal mass (C), and other parameters from Home Assistant history data.

## Home Assistant Integration
This repository includes a custom component for Home Assistant.
See [homeassistant/custom_components/housetemp/README.md](homeassistant/custom_components/housetemp/README.md) for installation and usage instructions.

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

## Testing & Plots

Unit tests are located in `tests/`.

```bash
python3 -m unittest tests/test_main.py
```

### Controlling Plots
By default, tests run silently without showing plots. To enable plots during tests (useful for debugging visual regressions), set the `SHOW_PLOTS` environment variable.

```bash
# Run tests with plots enabled
SHOW_PLOTS=1 python3 -m unittest tests/test_main.py
```
