"""Setup configuration for OpenGE package."""

from setuptools import setup, find_packages

setup(
    name="openge",
    version="0.1.0",
    description="Multi-crop trait prediction with G×E interaction modeling",
    author="OpenGE Contributors",
    author_email="info@openge.org",
    url="https://github.com/kirinhcl/openGE",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=5.4.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "interpretability": [
            "shap>=0.40.0",
            "captum>=0.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8+",
    ],
    include_package_data=True,
)
