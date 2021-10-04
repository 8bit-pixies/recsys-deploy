from distutils.core import setup

from setuptools import find_packages

REQUIRED = ["pandas", "scikit-learn", "colorama", "pydantic"]

DEV_REQUIRED = [
    "black",
    "pytest",
    "pytest-cov",
    "flake8",
    "isort",
    "mypy",  # to deal with later...
    "types-setuptools",
    "types-python-dateutil",
]

setup(
    name="recsys",
    version="0.1.0",
    description="recsys demo",
    author="charlie443",
    packages=find_packages(where="recsys"),
    install_requires=REQUIRED,
    extras_require={"dev": DEV_REQUIRED},
    include_package_data=True,
)
