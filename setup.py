from setuptools import setup, find_packages

setup(
    name="housetemp",
    version="0.1.0",
    description="Physics-based home thermal model",
    author="Daniel Sievers",
    packages=find_packages(include=["housetemp", "housetemp.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
    ],
)
