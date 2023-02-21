from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

setup(
    name="MicromagneticAnalysisTools",
    version="0.0.11",
    description="A collection of tools to analyse data from MuMax3.",
    author="Ross Knapman",
    author_email="rjknapman@gmail.com",
    packages=find_packages(include=['MicromagneticAnalysisTools', 'MicromagneticAnalysisTools/*']),
    install_requires=install_requires,
)