import importlib
from setuptools import setup, find_packages
import os

setup(
    name="stsrl",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "stsrl": ["slaythespire*.pyd"],
    }
)