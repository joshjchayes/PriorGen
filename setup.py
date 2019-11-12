'''
setup script for PriorGen

'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="priorgen",
    version="0.1.3",
    author="Joshua Hayes",
    author_email="joshjchayes@gmail.com",
    description="A package for Bayesian retrieval using machine learning generated priors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshjchayes/PriorGen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'dynesty', 'sklearn', 'scipy']
)
