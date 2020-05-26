import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='EOMcircuit', # Replace with your own username
    version="0.9",
    author="Raphaël Lescanne",
    author_email="raphael.lescanne@gmail.com",
    description="Solving the equations of motion of a superconducting circuit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rlescanne/EOMcircuit",
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'matplotlib', 'scipy'],
)