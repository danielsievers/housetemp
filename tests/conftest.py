"""Pytest configuration."""
import pytest
import sys
import os

# Add the 'homeassistant' directory to sys.path so we can import 'custom_components'
# This allows `from custom_components.housetemp import ...` to work if needed,
# or `homeassistant.custom_components.housetemp`.
# The plugin sets up a hass instance.

# We need to ensure that the code under `homeassistant/custom_components` is discoverable.
# We insert at 0 to take precedence over site-packages (which might have a custom_components namespace)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../homeassistant')))

@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test directory."""
    yield
