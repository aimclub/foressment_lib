from setuptools import setup
from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README_en.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="foressment_ai",
    version="0.1.3",
    description="Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://foressment_ai.readthedocs.io/",
    author="ComSec Lab",
    author_email="labcomsec@gmail.com",
    license="GNU GPLv3",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent"
    ],
    packages=["foressment_ai"],
    include_package_data=True,
    install_requires=[
        "scikit-learn", "scipy", "numpy",
        "pandas", "matplotlib", "keras",
        "tensorflow", "optuna", "psutil",
        "setuptools", "tqdm", "termcolor"
    ]
)
