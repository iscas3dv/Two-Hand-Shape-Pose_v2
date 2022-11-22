# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE https://github.com/facebookresearch/InterHand2.6M/blob/main/LICENSE
#

import os
import sys

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

