from setuptools import setup, find_packages
import sys
import os

sys.path.insert(0, os.path.abspath(__file__ + "/.."))
with open(os.path.abspath(__file__ + "/../lobio/__init__.py"), "r") as f:
    for line in f.readlines():
        if "__version__" in line:
            exec(line)
            break

setup(
    name="LOBio",
    version=__version__,
    author="Bogchamp & OdinManiac",
    description="Code for backtesting MM strategies.",
    url="https://gitflic.ru/project/aidynamicaction/rcognita",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Ubuntu",
    ],
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.1",
        "pandas==2.1.1",
        "tqdm==4.66.1",
        "matplotlib==3.8.0",
        "websockets==11.0.3",
    ],
    python_requires=">=3.11.0, <=3.12.0",
)
