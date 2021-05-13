#!/usr/bin/env python
"""
This is a main script to perform training or prediction of network on provided data.
To be called as main.py -conf conf_filename -a train
"""


import os
import copy
from argparse     import ArgumentParser
from argparse     import Namespace

from next_sparseconvnet.utils.data_loaders     import LabelType
from next_sparseconvnet.networks.architectures import NetArchitecture

def is_valid_action(parser, arg):
    if not arg in ['train']:#, 'predict']:
        parser.error("The action %s is not allowed!" % arg)
    else:
        return arg

def is_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def get_params(confname):
    full_file_name = os.path.expandvars(confname)
    parameters = {}

    builtins = __builtins__.__dict__.copy()
    #add enum classes
    builtins['LabelType']       = LabelType
    builtins['NetArchitecture'] = NetArchitecture

    with open(full_file_name, 'r') as config_file:
        exec(config_file.read(), {'__builtins__':builtins}, parameters)
    return Namespace(**parameters)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description="parameters for models")
    parser.add_argument("-conf", dest = "confname", required=True,
                        help = "input file with parameters", metavar="FILE",
                        type = lambda x: is_file(parser, x))
    parser.add_argument("-a", dest = "action" , required = True,
                        help = "action to do for NN",
                        type = lambda x : is_valid_action(parser, x))
    args     = parser.parse_args()
    confname = args.confname
    action   = args.action
    parameters = get_params(confname)

    # make sure LabelType and NetArchitectures are consistent
    # construct the network
    # if action is train:
        #train netowrk
    # else if prediction : still not implemented
