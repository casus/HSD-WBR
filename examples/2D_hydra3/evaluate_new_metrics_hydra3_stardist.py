from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
warnings.filterwarnings("ignore")

import sys
import click
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt


import os
from glob import glob
import time
from tqdm import tqdm
from tifffile import imread

from csbdeep.utils import Path

from stardist import gputools_available

# To run below command first do 'CC=gcc-11 CXX=g++-11 pip install -e .' from within the main code directory
from stardist.src.utils.hydranet import load_and_preprocess_data_eval_hydra3

from stardist.src.utils.config import read_json_config

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    parameters = config["parameters"]

    np.random.seed(int(parameters["seed"]))

    Y1_tst = sorted(glob(config["test"]["mask1_dir"]+'*'+parameters["extension"]))
    Y2_tst = sorted(glob(config["test"]["mask2_dir"]+'*'+parameters["extension"]))
    Y3_tst = sorted(glob(config["test"]["mask3_dir"]+'*'+parameters["extension"]))

    Y1_tst_pred = sorted(glob(config["test_pred"]["mask1_dir"]+'*'+parameters["extension"]))
    Y2_tst_pred = sorted(glob(config["test_pred"]["mask2_dir"]+'*'+parameters["extension"]))
    Y3_tst_pred = sorted(glob(config["test_pred"]["mask3_dir"]+'*'+parameters["extension"]))

    assert all(Path(y1).stem==Path(y2).stem==Path(y3).stem==(Path(y1_p).name.split("_pred")[0])==
               (Path(y2_p).name.split("_pred")[0])==(Path(y3_p).name.split("_pred")[0])
                for y1,y2,y3,y1_p,y2_p,y3_p in zip(Y1_tst,Y2_tst,Y3_tst,Y1_tst_pred,Y2_tst_pred,Y3_tst_pred))

    # Process test data
    Y1_tst, Y2_tst, Y3_tst = load_and_preprocess_data_eval_hydra3(Y1=Y1_tst,Y2=Y2_tst,Y3=Y3_tst)
    # Process predicted data
    Y1_tst_pred, Y2_tst_pred, Y3_tst_pred = load_and_preprocess_data_eval_hydra3(Y1=Y1_tst_pred,Y2=Y2_tst_pred, Y3=Y3_tst_pred)

    # Assembling configuration details for training
    use_gpu = parameters["use_gpu"] and gputools_available()

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8)
        # limit_gpu_memory(None, allow_growth=True) 

    def calculate_numerator_joint_tp_hydra3_dne(y1,y2,y3,y1_p,y2_p,y3_p):
        sem_y1_p = np.where(y1_p>0,1,0) # circle
        sem_y2 = np.where(y2>0,1,0) # rounded rect
        sem_y2_p = np.where(y2_p>0,1,0) 
        sem_y3_p = np.where(y3_p>0,1,0) # star

        return len(np.unique(np.multiply(
                    np.multiply(y1, sem_y1_p),
                    np.multiply(y3, sem_y3_p),
                    np.multiply(sem_y2, sem_y2_p))))-1

    def calculate_numerator_joint_tp_hydra3_de(y1,y2,y3,y1_p,y2_p,y3_p):
        sem_y1_p = np.where(y1_p>0,1,0) # circle
        sem_y2 = np.where(y2>0,1,0) # rounded rect
        sem_y2_p = np.where(y2_p>0,1,0) 
        sem_y3_p = np.where(y3_p>0,1,0) # star

        return len(np.unique(np.multiply(
                    np.multiply(y1, sem_y1_p),
                    np.multiply(sem_y2, sem_y2_p))))-1 + len(np.unique(np.multiply(
                         np.multiply(y3, sem_y3_p),
                         np.multiply(sem_y2, sem_y2_p))))-1

    denominator = [2*np.max(Y2_tst[id]) for id in range(len(Y2_tst))]

    if parameters['data_type']=='dne':
        numerator_tp = [calculate_numerator_joint_tp_hydra3_dne(y1,y2,y3,y1_p,y2_p,y3_p) 
                    for y1,y2,y3,y1_p,y2_p,y3_p in zip(Y1_tst,Y2_tst,Y3_tst,Y1_tst_pred,Y2_tst_pred,Y3_tst_pred)]
    elif parameters['data_type']=='de':             
        numerator_tp = [calculate_numerator_joint_tp_hydra3_de(y1,y2,y3,y1_p,y2_p,y3_p) 
                    for y1,y2,y3,y1_p,y2_p,y3_p in zip(Y1_tst,Y2_tst,Y3_tst,Y1_tst_pred,Y2_tst_pred,Y3_tst_pred)]
    else:
        print("Unknown data type specified in the config file.")

    print("New TP wrt channel 2:",np.mean(np.divide(numerator_tp,denominator))) #JTPR Value 

if __name__ == "__main__":
    main()