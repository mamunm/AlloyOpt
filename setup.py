from setuptools import find_packages, setup

setup(
    name="alloy_opt",
    version="1.0.0",
    packages=find_packages(include=["alloy_opt", "alloy_opt.*"]),
)
