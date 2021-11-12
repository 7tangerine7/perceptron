from setuptools import setup, find_packages

setup(
    name="perceptron",
    version="1.0",
    author="Polina",
    description="Little death. No README. No LICENSE.",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: what is license?",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "matplotlib"],
    entry_points={"console_scripts": ["perceptron = perceptron-code.main:main"]},
)
