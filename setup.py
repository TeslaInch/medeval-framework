"""
setup.py
~~~~~~~~
Packaging configuration for the ``medeval`` evaluation framework.

Allows local editable installation with:
    pip install -e .              # core only (numpy)
    pip install -e ".[nlp]"      # adds transformers, evaluate, datasets, torch
    pip install -e ".[dev]"      # adds pytest + pytest-cov

For production distribution, prefer pyproject.toml (PEP 517/518); this
setup.py is provided for maximum toolchain compatibility.
"""

from setuptools import find_packages, setup

# Read the long description from README.md so PyPI renders it correctly.
try:
    with open("README.md", encoding="utf-8") as fh:
        long_description: str = fh.read()
except FileNotFoundError:
    long_description = (
        "medeval: A rigorous open-source benchmarking framework for medical LLMs."
    )

setup(
    name="medeval",
    version="0.1.0",
    author="medeval contributors",
    description=(
        "An open-source Python framework for rigorously benchmarking medical LLMs "
        "for accuracy, hallucination rates, and clinical safety (ECE)."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/medeval-framework",
    license="MIT",
    # Automatically discover all sub-packages inside the ``medeval`` directory.
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"medeval": ["py.typed"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        # Full NLP stack for SemanticSimilarityScorer, NLIHallucinationDetector,
        # and BenchmarkLoader.
        "nlp": [
            "transformers>=4.35.0",
            "evaluate>=0.4.0",
            "datasets>=2.14.0",
            "torch>=2.0.0",
        ],
        # Developer / test tools.
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        # Convenience meta-extra: everything.
        "all": [
            "transformers>=4.35.0",
            "evaluate>=0.4.0",
            "datasets>=2.14.0",
            "torch>=2.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Typing :: Typed",
    ],
    keywords="medical llm evaluation benchmarking calibration ece hallucination sickle-cell",
)
