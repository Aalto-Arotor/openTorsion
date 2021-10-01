import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opentorsion",
    version="0.0.5",
    author="Aalto ARotor",
    author_email="todo@aalto.fi",
    description="Open source library for creating torsional finite element models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aalto-Arotor/openTorsion",
    project_urls={
        "Bug Tracker": "https://github.com/Aalto-Arotor/openTorsion/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"opentorsion": "opentorsion"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy"
    ]
)
