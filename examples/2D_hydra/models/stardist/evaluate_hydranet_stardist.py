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
import tensorflow as tf
import time
from tqdm import tqdm
from tifffile import imread

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
            plot_metrics_vs_tau, plot_individual_channel_predictions_hydra, example_hydra

from stardist.src.utils.config import read_json_config

lbl_cmap = random_label_cmap()
plot_model = keras_import('utils','plot_model')

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path, ):
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

    # Creating log directory
    model_dir = 'stardist' + '/' + f'{running_time}' 
    log_dir = model_dir + '_eval'
    
    
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

    # # Evaluating performance
    X_tst, Y1_tst, Y2_tst = load_and_preprocess_data_hydra(X=X_tst,Y1=Y1_tst,Y2=Y2_tst)

    pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False,
                            prob_thresh1=model.thresholds1['prob'], prob_thresh2=model.thresholds2['prob'],
                            nms_thresh1=model.thresholds1['nms'], nms_thresh2=model.thresholds2['nms']) for x in tqdm(X_tst)]

    Y2_tst_pred = [p[0][0] for p in pred]
    Y1_tst_pred = [p[1][0] for p in pred]

    # Plot example prediction
    plot_img_label_hydra(X_tst[0],Y1_tst[0], Y2_tst[0], lbl_title="label GT")
    path = Path(config["results_dir"]+log_dir+'/plots/'+'example_data_set.svg')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plot_img_label_hydra(X_tst[0],Y1_tst_pred[0],Y2_tst_pred[0], lbl_title="label Pred")
    path = Path(config["results_dir"]+log_dir+'/plots/'+'example_predicted_set.svg')
    plt.savefig(path)

    # Metrics for different taus
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats1 = [matching_dataset(Y1_tst, Y1_tst_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    stats2 = [matching_dataset(Y2_tst, Y2_tst_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

    print(stats1)
    print(stats2)

    # Plot metrics for different taus
    plot_metrics_vs_tau(stats=stats1,branch=1,log_dir=log_dir, results_dir=config["results_dir"])
    plot_metrics_vs_tau(stats=stats2,branch=2,log_dir=log_dir, results_dir=config["results_dir"])

    path = Path(config["results_dir"]+log_dir+'/stats1.txt')
    with open(path, 'w') as out_file:
        out_file.write("\n".join([str(stat) for stat in stats1]))

    path = Path(config["results_dir"]+log_dir+'/stats2.txt')
    with open(path, 'w') as out_file:
        out_file.write("\n".join([str(stat) for stat in stats2]))

    # Making predictions
    axis_norm = (0,1)
    img = normalize(X_tst[0], 1,99.8, axis=axis_norm)
    labels1,labels2 = model.predict_instances(img,prob_thresh1=model.thresholds1['prob'], prob_thresh2=model.thresholds2['prob'],
                                nms_thresh1=model.thresholds1['nms'], nms_thresh2=model.thresholds2['nms'])

    
    plot_individual_channel_predictions_hydra(img=img,labels=labels1)
    plot_individual_channel_predictions_hydra(img=img,labels=labels2)

    # Saving example predictions
    example_hydra(model=model,X=X_tst,Y1=Y1_tst,Y2=Y2_tst,i=0,log_dir=log_dir, results_dir=config["results_dir"])
    example_hydra(model=model,X=X_tst,Y1=Y1_tst,Y2=Y2_tst,i=1,log_dir=log_dir, results_dir=config["results_dir"])
    example_hydra(model=model,X=X_tst,Y1=Y1_tst,Y2=Y2_tst,i=2,log_dir=log_dir, results_dir=config["results_dir"])
    example_hydra(model=model,X=X_tst,Y1=Y1_tst,Y2=Y2_tst,i=3,log_dir=log_dir, results_dir=config["results_dir"])

if __name__ == "__main__":
    main()
