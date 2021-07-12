pl-grad-cam
================================

.. contents:: Table of Contents

Abstract
--------
ChRIS Plugin for Explainable AI visualization using the Grad-CAM algorithm

Setup
----
Download the relevant machine learning models whose results will be used as input
For example, for COVID-NET, download COVIDNet-CXR4-B from https://github.com/lindawangg/COVID-Net/blob/master/docs/models

Note
----
Grad-CAM largely depends on the provided reference model, so make sure that the model that is used to determine the result that is used as input exactly matches the provided reference model.

Acknowledgement
---------------
Jacob Gildenblat (jacobgil) for initial Grad-CAM implementation using keras: https://github.com/jacobgil/keras-grad-cam
