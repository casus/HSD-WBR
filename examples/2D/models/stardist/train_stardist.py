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
import shutil 
import tensorflow as tf
import time
from tqdm import tqdm
from tifffile import imread

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
        random_fliprot, random_intensity_change, augmenter, plot_metrics_vs_tau, \
            plot_individual_channel_predictions, example

from stardist.src.utils.config import read_json_config

lbl_cmap = random_label_cmap()
plot_model = keras_import('utils','plot_model')

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    parameters = config["parameters"]

    np.random.seed(int(parameters["seed"]))
    tf.random.set_seed(int(parameters["seed"]))

    branch = parameters["branch"]
    epochs = parameters["epochs"]
    quick_demo = parameters["quick_demo"]

    # Reading file names
    X = sorted(glob(config["train"]["img_dir"]+'*'+parameters["extension"]))
    Y = sorted(glob(config["train"]["mask_dir"]+'*'+parameters["extension"]))
    assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

    X_val = sorted(glob(config["validation"]["img_dir"]+'*'+parameters["extension"]))
    Y_val = sorted(glob(config["validation"]["mask_dir"]+'*'+parameters["extension"]))
    assert all(Path(x).name==Path(y).name for x,y in zip(X_val,Y_val))

    X_tst = sorted(glob(config["test"]["img_dir"]+'*'+parameters["extension"]))
    Y_tst = sorted(glob(config["test"]["mask_dir"]+'*'+parameters["extension"]))

    assert all(Path(x).name==Path(y).name for x,y in zip(X_tst,Y_tst))

    X_trn, Y_trn = X, Y
    X, Y = X[:10], Y[:10]

    # Reading images
    X = list(map(imread,X))
    Y = list(map(imread,Y))

    # Creating log directory
    running_time = time.strftime("%b-%d-%Y_%H-%M")
    log_dir = 'stardist' + '/' + f'{running_time}' 
    print(running_time)
    
    # Plotting example data
    i = min(4, len(X)-1)
    img, lbl = X[i], fill_label_holes(Y[i])
    assert img.ndim in (2,3)
    img = img if img.ndim==2 else img[...,:3]

    plt.figure(figsize=(16,10))
    plt.subplot(121); plt.imshow(img,cmap='gray');   plt.axis('off'); plt.title('Raw image')
    plt.subplot(122); plt.imshow(lbl,cmap=lbl_cmap); plt.axis('off'); plt.title('GT labels')
    path = Path(config["results_dir"]+log_dir+'/plots/'+'sample_data.svg')
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(path)
    
    # Showing reconstruction accuracy
    show_reconstruction_acc(Y=Y, branch=branch, log_dir=log_dir, results_dir=config["results_dir"])

    # Showing reconstruction polygon
    show_reconstruction_polygon(lbl=lbl, branch=branch, log_dir=log_dir, results_dir=config["results_dir"])

    # Process training data
    X_trn, Y_trn = load_and_preprocess_data(X=X_trn,Y=Y_trn)

    # Process validation data
    X_val, Y_val = load_and_preprocess_data(X=X_val,Y=Y_val)

    # Data checks
    assert len(X_trn) > 1, "not enough training data"
    assert len(X_val) > 1, "not enough training data"
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    # Printing example training data
    i = min(9, len(X_trn)-1)
    img, lbl = X_trn[i], Y_trn[i]
    assert img.ndim in (2,3)
    img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
    plot_img_label(img,lbl)
    None;
    path = Path(config["results_dir"]+log_dir+'/plots/'+'plot_img_label.svg')
    plt.savefig(path)

    # Assembling configuration details for training
    n_channel = 1 if X_trn[0].ndim == 2 else X_trn[0].shape[-1]
    n_rays = 32
    use_gpu = parameters["use_gpu"] and gputools_available()
    grid = (2,2)
    conf = Config2D (
        n_rays       = n_rays,
        grid         = grid,
        use_gpu      = use_gpu,
        n_channel_in = n_channel,
    )
    print(conf)
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # limit_gpu_memory(0.8)
        limit_gpu_memory(None, allow_growth=True)
        
    # Initialize model to train
    if parameters["finetune"]:
        # Initialize pretrained model
        model_pretrained = StarDist2D.from_pretrained('2D_versatile_he')
        shutil.copytree(model_pretrained.logdir, 
                        config["results_dir"]+log_dir, 
                        dirs_exist_ok=True)
        model = StarDist2D(None, name=log_dir, basedir=config["results_dir"])
    else:
        model = StarDist2D(conf, name=log_dir, basedir=config["results_dir"])

    # Check FOV
    check_fov(Y=Y_trn,model=model)

    # Plot augmented examples
    img, lbl = X_trn[0],Y_trn[0]
    plot_img_label(img, lbl)
    for i in range(3):
        img_aug, lbl_aug = augmenter(img,lbl)
        plot_img_label(img_aug, lbl_aug, img_title="augmented image", lbl_title=["similarly augmented label"])
        path = Path(config["results_dir"]+log_dir+'/plots/'+'augmented_set'+str(i)+'.svg')
        plt.savefig(path)

    # Save architecture diagram
    plot_model(model.keras_model, config["results_dir"]+log_dir+"/multi_input_and_output_model.png", show_shapes=True)

    # Train the model
    if quick_demo:
        print (
            "NOTE: This is only for a quick demonstration!\n"
            "      Please set the variable 'quick_demo = False' for proper (long) training.",
            file=sys.stderr, flush=True
        )
        model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                    epochs=2, steps_per_epoch=len(X_trn)//conf.train_batch_size)
    else:
        model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter,
                    epochs=epochs, steps_per_epoch=len(X_trn)//conf.train_batch_size)
    None;

    #Threshold Optimization
    if quick_demo:
        model.optimize_thresholds(X_val[:2], Y_val[:2])
    else:
        opt_n = min(len(X_val),175)
        model.optimize_thresholds(X_val[:opt_n], Y_val[:opt_n])

    # Evaluating performance
    # Process validation data
    X_tst, Y_tst = load_and_preprocess_data(X=X_tst,Y=Y_tst)

    Y_tst_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False,
                            prob_thresh=model.thresholds.prob, nms_thresh=model.thresholds.nms)[0] for x in tqdm(X_tst)]

    # Plot example prediction
    plot_img_label(X_tst[0],Y_tst[0], lbl_title=["label GT"])
    path = Path(config["results_dir"]+log_dir+'/plots/'+'example_data_set.svg')
    plt.savefig(path)
    plot_img_label(X_tst[0],Y_tst_pred[0], lbl_title=["label Pred"])
    path = Path(config["results_dir"]+log_dir+'/plots/'+'example_predicted_set.svg')
    plt.savefig(path)

    # Metrics for different taus
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(Y_tst, Y_tst_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

    print(stats)

    # Plot metrics for different taus
    plot_metrics_vs_tau(stats=stats,branch=branch,log_dir=log_dir, results_dir=config["results_dir"])

    # Load trained model
    if quick_demo:
        print (
            "NOTE: This is loading a previously trained demo model!\n"
            "      Please set the variable 'demo_model = False' to load your own trained model.",
            file=sys.stderr, flush=True
        )
        model = StarDist2D.from_pretrained('2D_demo')
    else:
        model = StarDist2D(None, name=log_dir, basedir=config["results_dir"])
    None;

    
    # Making predictions
    axis_norm = (0,1)
    img = normalize(X_tst[0], 0,100, axis=axis_norm)
    labels, details = model.predict_instances(img,prob_thresh=model.thresholds.prob,
                                nms_thresh=model.thresholds.nms)

    
    plot_individual_channel_predictions(img=img,labels=labels)

    # Saving example predictions
    example(model=model,X=X_tst,Y=Y_tst,i=0,log_dir=log_dir, results_dir=config["results_dir"])
    example(model=model,X=X_tst,Y=Y_tst,i=1,log_dir=log_dir, results_dir=config["results_dir"])
    example(model=model,X=X_tst,Y=Y_tst,i=2,log_dir=log_dir, results_dir=config["results_dir"])
    example(model=model,X=X_tst,Y=Y_tst,i=3,log_dir=log_dir, results_dir=config["results_dir"])

if __name__ == "__main__":
    main()