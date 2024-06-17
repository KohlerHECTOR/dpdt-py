from setuptools import setup, find_packages

__version__ = "0.1.1"


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="dpdt-py",
    version=__version__,
    author="Hector Kohler",
    author_email="hector.kohler@inria.fr",
    python_requires=">=3.8",
    packages=find_packages(),
    # package_data={'': extra_files},
    include_package_data=True,
    # package_dir={'':'src'},
    url="https://github.com/KohlerHECTOR/dpdt-py",
    description="Dynamic Programming Decision Tree",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["scikit-learn"],
)
