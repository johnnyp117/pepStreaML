from setuptools import setup, find_packages

# Dependencies
requirements = [
    "biopython",
    "deepspeed==0.5.9",
    "dm-tree",
    "pytorch-lightning",
    "omegaconf",
    "ml-collections",
    "einops",
    "scipy",
    "seaborn",
    "tqdm",
    "fair-esm",  # Added fair-esm package
]

setup(
    name='pepStreaML',
    version='0.3.0',  # Start with a small version number
    packages=find_packages(),  # Automatically discover and include all packages in the package directory
    install_requires=requirements,
    classifiers=[
        # Choose the right classifiers as per your project's needs
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # You can specify the minimum python version
)
