from setuptools import find_packages, setup

setup(
    name="falgen",
    version="0.1.0",
    packages=find_packages(),
    description="Fal's Generative Model Toolbox",
    author="Fal Research Team",
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "tqdm",
        "wandb",
        "click",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
