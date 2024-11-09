from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scalingfilter",
    version="0.1.0",
    author="Ruihang Li",
    author_email="ruihangli@mail.ustc.edu.cn",
    description="An official implementation for 'ScalingFilter: Assessing Data Quality through Inverse Utilization of Scaling Laws'.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scalingfilter/scalingfilter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "transformers",
        "more_itertools",
        "tqdm",
        "scipy<2",
        "sentence-transformers",
        "jsonlines",
        "numpy",
    ],
)