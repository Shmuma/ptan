"""
PTAN stands for PyTorch AgentNet -- reimplementation of AgentNet library for pytorch
"""
import pathlib
import setuptools


requirements = pathlib.Path("requirements.txt").read_text().splitlines()


setuptools.setup(
    name="ptan",
    author="Max Lapan",
    author_email="max.lapan@gmail.com",
    license='GPL-v3',
    description="PyTorch reinforcement learning framework",
    version="0.8",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
