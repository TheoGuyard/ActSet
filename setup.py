# encoding: utf-8

import os, pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
   name="turbo-screening",
   version="1.0.0",
   description="Active-set solver for non-negative LASSO problem",
   long_description=README,
   long_description_content_type="text/markdown",
   author="T. Guyard",
   author_email="guyard.theo@gmail.com",
   license="MIT",
   python_requires=">=3",
   classifiers=[
      "License :: OSI Approved :: MIT License",
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.7",
   ],
   packages=find_packages(),
   include_package_data=True,
)
