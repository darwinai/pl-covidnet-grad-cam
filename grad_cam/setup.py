import sys
import os

# Make sure we are running python3.5+
if 10 * sys.version_info[0] + sys.version_info[1] < 35:
    sys.exit("Sorry, only Python 3.5+ is supported.")

from setuptools import setup


def readme():
    print("Current dir = %s" % os.getcwd())
    print(os.listdir())
    with open('README.rst') as f:
        return f.read()


setup(
    name='grad_cam',
    # for best practices make this version the same as the VERSION class variable
    # defined in your ChrisApp-derived Python class
    version='0.1',
    description='Plugin to ChRIS for Grad-CAM functionalities',
    long_description=readme(),
    author='DarwinAI (Matthew Wang)',
    author_email='matthew.wang@darwinai.ca',
    url='https://github.com/darwinai/pl-grad-cam',
    packages=['grad_cam'],
    install_requires=['chrisapp', 'pudb'],
    test_suite='nose.collector',
    tests_require=['nose'],
    scripts=['grad_cam/grad_cam.py'],
    license='AGPL 3.0',
    zip_safe=False)
