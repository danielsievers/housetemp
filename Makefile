.PHONY: setup test clean

# Development environment setup (requires Python 3.13+)
setup:
	@echo "Creating virtual environment with Python 3.13..."
	python3.13 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	@echo ""
	@echo "✅ Setup complete! Activate with: source .venv/bin/activate"

# Run tests
test:
	.venv/bin/python -m pytest tests/ -v

# Run tests with coverage
test-cov:
	.venv/bin/python -m pytest tests/ --cov=custom_components/housetemp --cov-report=term-missing

# Clean build artifacts
clean:
	rm -rf .venv
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -f *.pyc */*.pyc */*/*.pyc
