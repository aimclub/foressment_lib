from setuptools import setup

from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="aopssop",
    version="0.1.0",
    description="Demo library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://aopssop.readthedocs.io/",
    author="ComSec Lab",
    author_email="labcomsec@gmail.com",
    license="GNU GPLv3",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["aopssop"],
    include_package_data=True,
    install_requires=[
        "sklearn", "scipy", "numpy",
        "pandas", "matplotlib", "keras",
        "tensorflow"
    ]
)
