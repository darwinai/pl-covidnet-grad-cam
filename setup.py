from os import path

from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.rst')) as f:
    readme = f.read()

setup(
    name='covidnet_grad_cam',
    # for best practices make this version the same as the VERSION class variable
    # defined in your ChrisApp-derived Python class
    version='1.0.0',
    description='Plugin to ChRIS for Grad-CAM functionalities',
    long_description=readme,
    author='DarwinAI (Matthew Wang)',
    author_email='matthew.wang@darwinai.ca',
    url='https://github.com/darwinai/pl-covidnet-grad-cam',
    packages=['covidnet_grad_cam'],
    install_requires=['chrisapp', 'pudb'],
    test_suite='nose.collector',
    tests_require=['nose'],
    python_requires='>=3.6',
    entry_points={
        'console_scripts':
        ['covidnet-grad-cam = covidnet_grad_cam.__main__:main']
    },
    license='AGPL 3.0',
    zip_safe=False)
