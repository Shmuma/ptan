"""
PTAN stands for PyTorch AgentNet -- reimplementation of AgentNet library for pytorch
"""
import setuptools


requirements = ['torch==1.13.0', 'gymnasium', 'atari-py', 'numpy', 'opencv-python']


setuptools.setup(
    name="ptan",
    author="Max Lapan",
    author_email="max.lapan@gmail.com",
    license='GPL-v3',
    description="PyTorch reinforcement learning framework",
    version="1.0",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
