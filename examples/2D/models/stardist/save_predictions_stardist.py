from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
warnings.filterwarnings("ignore")

import sys
import click
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
from pathlib import Path

import os
from glob import glob
import tensorflow as tf
import time
from tqdm import tqdm
from tifffile import imread, imwrite

from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.utils import Path, normalize, download_and_extract_zip_file

from stardist import random_label_cmap,fill_label_holes, relabel_image_stardist, \
                    calculate_extents, gputools_available, _draw_polygons, export_imagej_rois
from stardist.matching import matching_dataset, matching

# To correctly use below command comment line 4 and uncomment line 3 in stardist > models > __init__.py
from stardist.models import Config2D, StarDist2D, StarDistData2D

# To run below command first do 'CC=gcc-11 CXX=g++-11 pip install -e .' from within the main code directory
from stardist.src.utils.tf import keras_import
from stardist.src.utils.hydranet import show_reconstruction_acc, \
    show_reconstruction_polygon, load_and_preprocess_data, plot_img_label, check_fov, \
        random_fliprot, random_intensity_change, augmenter, plot_metrics_vs_tau

from stardist.src.utils.config import read_json_config

lbl_cmap = random_label_cmap()
plot_model = keras_import('utils','plot_model')

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    running_time = config["running_time"]
    parameters = config["parameters"]

    np.random.seed(int(parameters["seed"]))
    tf.random.set_seed(int(parameters["seed"]))

    quick_demo = parameters["quick_demo"]

    X_tst = sorted(glob(config["test"]["img_dir"]+'*'+parameters["extension"]))
    Y_tst = sorted(glob(config["test"]["mask_dir"]+'*'+parameters["extension"]))

    assert all(Path(x).name==Path(y).name for x,y in zip(X_tst,Y_tst))

    Y_tst_names =  Y_tst
    
    # Process test data
    X_tst, Y_tst = load_and_preprocess_data(X=X_tst,Y=Y_tst)


    # Creating log directory
    model_dir = 'stardist' + '/' + f'{running_time}'  
    log_dir = model_dir + '_pred'
    Path(config["results_dir"]+log_dir).mkdir(parents=True,exist_ok=True)

    # Assembling configuration details for training
    use_gpu = parameters["use_gpu"] and gputools_available()

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8)
        # limit_gpu_memory(None, allow_growth=True)
        

    # Load trained model
    if quick_demo:
        print (
            "NOTE: This is loading a previously trained demo model!\n"
            "      Please set the variable 'demo_model = False' to load your own trained model.",
            file=sys.stderr, flush=True
        )
        model = StarDist2D.from_pretrained('2D_demo')
    else:
        model = StarDist2D(None, name=model_dir, basedir=config["results_dir"])
    None;

    print(model.keras_model.summary())

    # Saving predictions

    Y_tst_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), 
                show_tile_progress=False,
                prob_thresh=model.thresholds.prob, nms_thresh=model.thresholds.nms)[0] 
                for x in tqdm(X_tst)]
    print("Pred Min, Max :", np.min(Y_tst_pred[0]),np.max(Y_tst_pred[0]))

    if config["test"]["start_idx"]:
        _ = [imwrite(Path(config["results_dir"]+log_dir+"/"+
                    str(int(config["test"]["start_idx"])+id)+'_pred'+parameters["extension"]), 
                    Y_tst_pred[id].astype(np.uint8)) for id in range(len(Y_tst_pred))]
    else:
        _ = [imwrite(Path(config["results_dir"]+log_dir+"/"+
                    Path(Y_tst_names[id]).stem+'_pred'+parameters["extension"]), 
                    Y_tst_pred[id].astype(np.uint8)) for id in range(len(Y_tst_pred))]

if __name__ == "__main__":
    main()