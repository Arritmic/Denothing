import setuptools
import os

requirementPath = 'requirements.txt'
reqs = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        reqs = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Denothing",
    version="0.0.1",
    author="Constantino Álvarez Casado <constantino.alvarezcasado@oulu.fi>",
    author_email="",
    description="Package for SSL training of convolutional AE for denoising.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arritmic/Denothing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approvedconda  :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=reqs
)
