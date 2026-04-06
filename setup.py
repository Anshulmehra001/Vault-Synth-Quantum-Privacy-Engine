from setuptools import setup, find_packages

setup(
    name="vault-synth",
    version="0.1.0",
    description="Quantum Privacy Engine for Synthetic Financial Data Generation",
    author="Vault-Synth Team",
    packages=find_packages(),
    install_requires=[
        "pennylane>=0.30.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "dev": [
            "hypothesis>=6.70.0",
            "pytest>=7.2.0",
        ]
    },
    python_requires=">=3.8",
)
