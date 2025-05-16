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
# To correctly use below command comment line 3 and uncomment line 4 in stardist > models > __init__.py
from stardist.models import Config2D, StarDist2D, StarDistData2D

# To run below command first do 'CC=gcc-11 CXX=g++-11 pip install -e .' from within the main code directory
from stardist.src.utils.tf import keras_import
from stardist.src.utils.hydranet import show_reconstruction_acc, \
    show_reconstruction_polygon, load_and_preprocess_data_hydra, plot_img_label_hydra, \
        check_fov,  random_fliprot_hydra, random_intensity_change, augmenter_hydra, \
            plot_metrics_vs_tau

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
    Y1_tst = sorted(glob(config["test"]["mask1_dir"]+'*'+parameters["extension"]))
    Y2_tst = sorted(glob(config["test"]["mask2_dir"]+'*'+parameters["extension"]))
    assert all(Path(x).name==Path(y1).name==Path(y2).name for x,y1,y2 in zip(X_tst,Y1_tst,Y2_tst))


    Y1_tst_names = Y1_tst
    Y2_tst_names = Y2_tst

    # Creating log directory
    model_dir = 'stardist' + '/' + f'{running_time}' 
    log_dir = model_dir + '_pred'
    Path(config["results_dir"]+log_dir).mkdir(parents=True,exist_ok=True)
    if config["test"]["start_idx"]:
        Path(config["results_dir"]+log_dir+"/pred_cyt/").mkdir(parents=True,exist_ok=True)
        Path(config["results_dir"]+log_dir+"/pred_nuc/").mkdir(parents=True,exist_ok=True)
    else:
        Path(config["results_dir"]+log_dir+"/pred_well/").mkdir(parents=True,exist_ok=True)
        Path(config["results_dir"]+log_dir+"/pred_plaque/").mkdir(parents=True,exist_ok=True)

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
    X_tst, Y1_tst, Y2_tst = load_and_preprocess_data_hydra(X=X_tst,Y1=Y1_tst,Y2=Y2_tst)

    pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False,
                            prob_thresh1=model.thresholds1['prob'], prob_thresh2=model.thresholds2['prob'],
                            nms_thresh1=model.thresholds1['nms'], nms_thresh2=model.thresholds2['nms']) for x in tqdm(X_tst)]

    Y2_tst_pred = [p[0][0] for p in pred]
    Y1_tst_pred = [p[1][0] for p in pred]

    print("Pred Y1 Min, Max :", np.min(Y1_tst_pred[0]),np.max(Y1_tst_pred[0]))
    print("Pred Y2 Min, Max :", np.min(Y2_tst_pred[0]),np.max(Y2_tst_pred[0]))

    if config["test"]["start_idx"]:
        _ = [imwrite(Path(config["results_dir"]+log_dir+"/pred_nuc/"+
                    str(int(config["test"]["start_idx"])+id)+'_pred'+parameters["extension"]), 
                    Y1_tst_pred[id].astype(np.uint8)) for id in range(len(Y1_tst_pred))]

        _ = [imwrite(Path(config["results_dir"]+log_dir+"/pred_cyt/"+
                    str(int(config["test"]["start_idx"])+id)+'_pred'+parameters["extension"]), 
                    Y2_tst_pred[id].astype(np.uint8)) for id in range(len(Y2_tst_pred))]
    else:
        _ = [imwrite(Path(config["results_dir"]+log_dir+"/pred_plaque/"+
                    Path(Y1_tst_names[id]).stem+'_pred'+parameters["extension"]), 
                    Y1_tst_pred[id].astype(np.uint8)) for id in range(len(Y1_tst_pred))]

        _ = [imwrite(Path(config["results_dir"]+log_dir+"/pred_well/"+
                    Path(Y2_tst_names[id]).stem+'_pred'+parameters["extension"]), 
                    Y2_tst_pred[id].astype(np.uint8)) for id in range(len(Y2_tst_pred))]
    
if __name__ == "__main__":
    main()