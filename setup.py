from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pydpdt',
    version='0.1.0',
    author='Hector Kohler',
    author_email='hector.kohler@inria.fr',
    description='Dynamic Programming Decision Tree',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KohlerHECTOR/pydpdt',  # Replace with your project's URL
    packages=find_packages(),  # Automatically find packages in your project
    include_package_data=True,
    install_requires=required,  # List of dependencies
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change the license if needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)