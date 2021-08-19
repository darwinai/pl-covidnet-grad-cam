pl-grad-cam
================================

.. contents:: Table of Contents

Abstract
--------
ChRIS Plugin for Explainable AI visualization using the Grad-CAM algorithm

Synopsis
--------

.. code:: python

    python grad_cam.py                                         \\
    [-h] [--help]                                               \\                                                 
    [--man]                                                     \\
    [--meta]                                                    \\                                       
    [-v <level>] [--verbosity <level>]                          \\
    [--version]                                                 \\
    [--modelname <modelname>]                                   \\
    --imagefile <imagefile>                                   \\ 
    --predmatrix <predmatrix>                                   \\
    <inputDir>                                                  \\
    <outputDir>


Arguments
---------

.. code::

    [-h] [--help]
    If specified, show help message and exit.

    [--man]
    If specified, print (this) man page and exit.

    [--meta]
    If specified, print plugin meta data and exit.

    [-v <level>] [--verbosity <level>]
    Verbosity level for app. Not used currently.

    [--version]
    If specified, print version number and exit. 

    [--modelname]
    The name of the model being used, this is optional (default is COVIDNET).

    [--imagefile]
    The name of the input image in the input directory, this is required

    [--predmatrix]
    The name of the prediction matrix file in the input directory, this is required


Setup
-----

Download the relevant machine learning models whose results will be used as input
For example, for COVID-NET, download COVIDNet-CXR4-B from https://github.com/lindawangg/COVID-Net/blob/master/docs/models.md

Then, put the downloaded folder(s) in grad_cam/models

The folder structure should be:

pl-grad-cam/grad_cam/models/COVIDNet-CXR4-B

Run
---

.. code:: bash

    cd grad_cam
    python grad_cam.py --imagefile ex-covid.jpg --predmatrix raw-prediction-matrix-default.json ../in ../out

ex-covid.jpg is the name of the input image in the input directory

`in/` is the input directory, and its relative path from `grad_cam/` is `../in`

`out/` is the output directory, and its relative path from `grad_cam/` is `../out`

Using ``docker run``
~~~~~~~~~~~~~~~~~~~~

To run using ``docker``, be sure to assign an "in" directory to ``/incoming`` and an "out" directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

Start from the pl-grad_cam directory

build the container using

.. code:: bash

    docker build -t local/pl-grad-cam .

Now, run the container:

.. code:: bash

    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing                       \
            pl-grad-cam grad_cam --imagefile ex-covid.jpg --predmatrix raw-prediction-matrix-default.json /incoming /outgoing                       \

This is bind mounting the in and out directory under pl-grad-cam. Feel free to create different directories.

Make sure the input directory contain an image that fits the --imagefile argument, and make sure the incoming and outgoing directories used as input are the ones being bind mounted.

You can create different directories using the following command. chmod 777 out just makes out directory writable

.. code:: bash

    mkdir in out && chmod 777 out

Note
----
Grad-CAM largely depends on the provided reference model, so make sure that the model that is used to determine the result that is used as input exactly matches the provided reference model.

Acknowledgement
---------------
Insik Kim(insikk) for initial Grad-CAM implementation for ResNet and VGG using tensorflow: https://github.com/insikk/Grad-CAM-tensorflow
