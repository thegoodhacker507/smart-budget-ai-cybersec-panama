
"""
Smart Budget AI - Setup Script
Script de configuración para instalación del paquete
"""

from setuptools import setup, find_packages
import os

# Leer el archivo README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Leer requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="smart-budget-ai",
    version="1.0.0",
    author="Smart Budget AI Team",
    author_email="contact@smartbudgetai.com",
    description="Sistema de Recomendaciones de Presupuesto Inteligente con Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/smart-budget-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
        ],
        "prod": [
            "gunicorn>=20.0",
            "nginx",
        ]
    },
    entry_points={
        "console_scripts": [
            "smart-budget-train=src.train:main",
            "smart-budget-app=app.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.html", "*.css", "*.js", "*.md", "*.txt"],
    },
    zip_safe=False,
)
