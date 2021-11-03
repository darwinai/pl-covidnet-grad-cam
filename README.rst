pl-covidnet-grad-cam
====================

.. contents:: Table of Contents

Abstract
--------
ChRIS Plugin for Explainable AI visualization using the Grad-CAM algorithm

Synopsis
--------

.. code::

    python grad_cam.py                           \
        [-h] [--help]                            \
        [--man]                                  \
        [--meta]                                 \
        [-v <level>] [--verbosity <level>]       \
        [--version]                              \
        [--modelname <modelname>]                \
        --imagefile <imagefile>                  \
        --predmatrix <predmatrix>                \
        <inputDir>                               \
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
    The name of the model being used, this is optional (default is COVIDNet-CXR4-B).

    [--imagefile]
    The name of the input image in the input directory, this is required

    [--predmatrix]
    The name of the prediction matrix file in the input directory, this is required


Local Build
-----------

.. code:: bash

    DOCKER_BUILDKIT=1 docker build -t local/pl-covidnet-grad-cam .

Run
---

.. code:: bash

    docker run --rm -v $PWD/in:/incoming -v $PWD/out:/outgoing                       \
        darwinai/covidnet-grad-cam-pl covidnet-grad-cam                              \
            --imagefile ex-covid.jpg --predmatrix raw-prediction-matrix-default.json \
            /incoming /outgoing


Models
------

The COVIDNet-CXR4-B model is downloaded from https://drive.google.com/drive/folders/1i5XxVy6A6Dwn0IIoGqpbvQo3xgWlVgB_
For more information, visit https://github.com/lindawangg/COVID-Net/blob/master/docs/models.md

Note
----
Grad-CAM largely depends on the provided reference model, so make sure that the model
that is used to determine the result that is used as input exactly matches the provided
reference model.

Acknowledgement
---------------
Insik Kim(insikk) for initial Grad-CAM implementation for ResNet and VGG using 
tensorflow: https://github.com/insikk/Grad-CAM-tensorflow
