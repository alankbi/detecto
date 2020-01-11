import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="detecto",
    version="1.0.1",
    author="Alan Bi",
    author_email="alan.bi326@gmail.com",
    description="A Python package for quick and easy object detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alankbi/detecto",
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib',
        'opencv-python',
        'pandas',
        'scikit-image',
        'torch',
        'torchvision',
        'pillow==6.2.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
