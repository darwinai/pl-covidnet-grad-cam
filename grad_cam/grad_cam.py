#!/usr/bin/env python
#
# Grad-CAM ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import os
import sys

sys.path.append(os.path.dirname(__file__))
from inference import Inference
# import the Chris app superclass
from chrisapp.base import ChrisApp

Gstr_title = """

Generate a title from 
http://patorjk.com/software/taag/#p=display&f=Doom&t=Grad-CAM

"""

Gstr_synopsis = """

(Edit this in-line help for app specifics. At a minimum, the 
flags below are supported -- in the case of DS apps, both
positional arguments <inputDir> and <outputDir>; for FS apps
only <outputDir> -- and similarly for <in> <out> directories
where necessary.)

    NAME

       grad_cam.py 

    SYNOPSIS

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

    BRIEF EXAMPLE

        * Bare bones execution

            mkdir in out && chmod 777 out
            python grad_cam.py   \\
                                in    out

    DESCRIPTION

        `grad_cam.py` ...

    ARGS

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


"""


class GradCam(ChrisApp):
    """
    Plugin to ChRIS for Grad-CAM functionalities.
    """
    AUTHORS = 'Matthew Wang, DarwinAI (matthew.wang@darwinai.ca)'
    SELFPATH = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC = os.path.basename(__file__)
    EXECSHELL = 'python3'
    TITLE = 'A ChRIS plugin app'
    CATEGORY = ''
    TYPE = 'ds'
    DESCRIPTION = 'Plugin to ChRIS for Grad-CAM functionalities'
    DOCUMENTATION = 'http://wiki'
    VERSION = '0.1'
    ICON = ''  # url of an icon image
    LICENSE = 'AGPL 3.0'
    MAX_NUMBER_OF_WORKERS = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS = 1  # Override with integer value
    MAX_CPU_LIMIT = ''  # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT = ''  # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT = ''  # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT = ''  # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument('--modelname',
                          dest='modelname',
                          type=str,
                          optional=True,
                          help='Name of covidnet model being used',
                          default='COVIDNet-CXR4-B')
        self.add_argument('--imagefile',
                          dest='imagefile',
                          type=str,
                          optional=False,
                          help='Name of image file to infer from')
        self.add_argument('--predmatrix',
                          dest='predmatrix',
                          type=str,
                          optional=False,
                          help='Name of file containing prediction matrix')
        self.add_argument(
            '--input_size',
            dest='input_size',
            type=int,
            optional=True,
            help='Size of input (ex: if 480x480, --input_size 480)',
            default=480)
        self.add_argument('--top_percent',
                          dest='top_percent',
                          type=float,
                          optional=True,
                          help='Percent top crop from top of image',
                          default=0.08)

    def add_model_to_options(self, options, model_info):
        options.weightspath = os.getcwd() + model_info['weightspath']
        options.ckptname = model_info['ckptname']
        options.metaname = model_info['metaname']
        options.in_tensor = model_info['in_tensor']
        options.out_softmax_tensor = model_info['out_softmax_tensor']
        options.labels_tensor = model_info['labels_tensor']
        options.sample_weights_tensor = model_info['sample_weights_tensor']
        options.out_logit_tensor = model_info['out_logit_tensor']
        options.last_conv_out_tensor = model_info['last_conv_out_tensor']
        return options

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())
        models = {
            'COVIDNet-CXR4-B': {
                'weightspath': '/models/COVIDNet-CXR4-B',
                'ckptname': 'model-1545',
                'metaname': 'model.meta',
                'in_tensor': 'input_1:0',
                'out_softmax_tensor': 'norm_dense_1/Softmax:0',
                'labels_tensor': 'norm_dense_1_target:0',
                'sample_weights_tensor': 'norm_dense_1_sample_weights:0',
                'out_logit_tensor': 'norm_dense_1/MatMul:0',
                'last_conv_out_tensor': 'conv5_block3_out/add:0'
            }
        }
        options = self.add_model_to_options(options, models[options.modelname])
        inference = Inference(options)
        inference.infer()

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = GradCam()
    chris_app.launch()

# chris app needs to write to files as outputs and taking inputs
# output a dicom image then ChRIS user interface will be able to show it
# csv, json, or custom html css files
