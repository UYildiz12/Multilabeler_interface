from setuptools import setup, find_packages

setup(
    name="multilabeler",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.29.0",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "requests>=2.31.0"
    ]
)