from setuptools import setup, find_packages

setup(
    name='CPII_RealEstate',
    version='0.1',
    # automatically pick up CPII_RealEstate and all its subpackages
    packages=find_packages(),
    include_package_data=True,        # if you add a MANIFEST.in for e.g. CSV files
    entry_points={
        'console_scripts': [
            # now points inside your package
            'predict-house=CPII_RealEstate.predict_from_input:main',
        ],
    },
    author='Daveed Vodonenko',
    description='From-scratch machine learning models to predict real estate prices.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)