from __future__ import division,print_function
import torchvision
import torch
import logging
import sys, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import radiomics
import six
from radiomics import featureextractor
from PIL import Image




# Get the PyRadiomics logger (default log-level = INFO)
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

data_path = '/home/harryzhang/Documents/athena_screen/images'

imageNames = []

files = os.listdir(path)
for name in files:
    imageNames.append(name)
