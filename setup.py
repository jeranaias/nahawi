#!/usr/bin/env python3
"""Setup script for Nahawi Arabic GEC."""

from setuptools import setup, find_packages

setup(
    name="nahawi",
    version="0.1.0",
    description="Arabic Grammatical Error Correction Ensemble",
    author="Nahawi Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "nahawi=nahawi_ensemble.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Text Processing :: Linguistic",
    ],
)
