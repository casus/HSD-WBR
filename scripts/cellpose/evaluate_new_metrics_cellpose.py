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
from stardist.src.utils.hydranet import load_and_preprocess_data_eval

from stardist.src.utils.config import read_json_config

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    parameters = config["parameters"]

    np.random.seed(int(parameters["seed"]))

    Y1_tst = sorted(glob(config["test"]["mask1_dir"]+'*'+parameters["extension"]))
    Y2_tst = sorted(glob(config["test"]["mask2_dir"]+'*'+parameters["extension"]))
    Y1_tst_pred = sorted(glob(config["test_pred"]["mask1_dir"]+'*'+parameters["extension"]))
    Y2_tst_pred = sorted(glob(config["test_pred"]["mask2_dir"]+'*'+parameters["extension"]))

    assert all(Path(y1).stem==Path(y2).stem==(Path(y1_p).name.split("_img_pred")[0])==(Path(y2_p).name.split("_img_pred")[0]) 
                for y1,y2,y1_p,y2_p in zip(Y1_tst,Y2_tst,Y1_tst_pred,Y2_tst_pred))

    # Process test data
    Y1_tst, Y2_tst = load_and_preprocess_data_eval(Y1=Y1_tst,Y2=Y2_tst)
    # Process predicted data
    Y1_tst_pred, Y2_tst_pred = load_and_preprocess_data_eval(Y1=Y1_tst_pred,Y2=Y2_tst_pred)

    # Assembling configuration details for training
    use_gpu = parameters["use_gpu"] and gputools_available()

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8)
        # limit_gpu_memory(None, allow_growth=True) 

    def calculate_numerator_fp(y1,y2,y1_p,y2_p):
        inv_sem_y1 = np.where(y1>0,0,1)
        sem_y2_p = np.where(y2_p>0,1,0)
        sem_y1_p = np.where(y1_p>0,1,0)
        return len(np.unique(np.multiply(
                    np.multiply(y2, sem_y2_p),
                    np.multiply(inv_sem_y1, sem_y1_p))))-1

    def calculate_numerator_fn(y1,y2,y1_p,y2_p):
        inv_sem_y1_p = np.where(y1_p>0,0,1)
        sem_y2_p = np.where(y1_p>0,1,0)
        sem_y1 = np.where(y1,1,0)
        return len(np.unique(np.multiply(
                    np.multiply(y2, sem_y2_p),
                    np.multiply(sem_y1, inv_sem_y1_p))))-1

    def calculate_numerator_joint_tp(y1,y2,y1_p,y2_p):
        sem_y1 = np.where(y1>0,1,0)
        sem_y1_p = np.where(y1_p>0,1,0)
        sem_y2_p = np.where(y2_p>0,1,0)
        return len(np.unique(np.multiply(
                    np.multiply(y2, sem_y2_p),
                    np.multiply(sem_y1, sem_y1_p))))-1

    def calculate_numerator_joint_tp_vacv(y1,y2,y1_p,y2_p):
        sem_y2 = np.where(y2>0,1,0)
        sem_y1_p = np.where(y1_p>0,1,0)
        sem_y2_p = np.where(y2_p>0,1,0)
        return len(np.unique(np.multiply(
                    np.multiply(sem_y2, sem_y2_p),
                    np.multiply(y1, sem_y1_p))))-1

    denominator = [np.max(Y1_tst[id]) for id in range(len(Y1_tst))]

    numerator_fp = [calculate_numerator_fp(y1,y2,y1_p,y2_p) 
                    for y1,y2,y1_p,y2_p in zip(Y1_tst,Y2_tst,Y1_tst_pred,Y2_tst_pred)] 

    numerator_fn = [calculate_numerator_fn(y1,y2,y1_p,y2_p) 
                    for y1,y2,y1_p,y2_p in zip(Y1_tst,Y2_tst,Y1_tst_pred,Y2_tst_pred)] 

    if parameters["vacv"]:
        numerator_tp = [calculate_numerator_joint_tp_vacv(y1,y2,y1_p,y2_p) 
                    for y1,y2,y1_p,y2_p in zip(Y1_tst,Y2_tst,Y1_tst_pred,Y2_tst_pred)]
    else:
        numerator_tp = [calculate_numerator_joint_tp(y1,y2,y1_p,y2_p) 
                    for y1,y2,y1_p,y2_p in zip(Y1_tst,Y2_tst,Y1_tst_pred,Y2_tst_pred)]

    print("New FP wrt channel 1:",np.mean(np.divide(numerator_fp,denominator)))
    print("New FN wrt channel 1:",np.mean(np.divide(numerator_fn,denominator)))
    print("New TP wrt channel 1:",np.mean(np.divide(numerator_tp,denominator))) #JTPR

if __name__ == "__main__":
    main()