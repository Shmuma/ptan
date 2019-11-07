"""
PTAN stands for PyTorch AgentNet -- reimplementation of AgentNet library for pytorch
"""
import setuptools


requirements = ['torch==1.3.0', 'gym', 'atari-py', 'numpy', 'opencv-python']


setuptools.setup(
    name="ptan",
    author="Max Lapan",
    author_email="max.lapan@gmail.com",
    license='GPL-v3',
    description="PyTorch reinforcement learning framework",
    version="0.6",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
