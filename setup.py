from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'torch',
    'torchvision',
    'python-box==5.3.0'
]

setup(
    name='reiforcement-learning',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.6",
    packages=find_packages(),
    include_package_data=True,
    author='Souranil Sen',
    description='Reinforcement Learning sample environments'
)