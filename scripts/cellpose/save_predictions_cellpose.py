import click
import sys
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path

import time
from tifffile import imwrite
from cellpose import io, models
from stardist import random_label_cmap
from stardist.matching import matching

from stardist.src.utils.config import read_json_config

lbl_cmap = random_label_cmap()

def example_cellpose(model, X, Y, i, log_dir, results_dir):
    Y_pred, _, _ = model.eval([X], channels=[1, 3])

    plt.figure(figsize=(13,10))
    img_show = X if X.ndim==2 else X[...,0]
    plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.subplot(121).set_title('prediction'); plt.axis('off')
    a = plt.axis()
    plt.imshow(Y_pred[0].astype(np.uint32), cmap=lbl_cmap, alpha=0.5)
    plt.axis(a)
    plt.subplot(122); plt.imshow(img_show, cmap='gray'); plt.subplot(122).set_title('ground truth'); plt.axis('off')
    plt.imshow(Y.astype(np.uint32), cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    path = Path(results_dir+log_dir+'/plots/'+'prediction_overlayed_'+str(i)+'.svg')
    print(path)
    plt.savefig(path)

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    if config["running_time"]:
        running_time = config["running_time"]
    else:
        running_time = time.strftime("%b-%d-%Y_%H-%M")
    parameters = config["parameters"]

    np.random.seed(int(parameters["seed"]))

    # Creating log directory
    model_dir = 'cellpose' + '/' + f'{running_time}'
    log_dir = model_dir + '_pred'
    Path(config["results_dir"]+log_dir).mkdir(parents=True,exist_ok=True)

    TRAIN_DIR =  parameters["train_dir"]
    TEST_DIR = parameters["test_dir"]
    IMG_FILTER = parameters["img_filter"]
    MASK_FILTER = parameters["mask_filter"]
    LOOK_ONE_LEVEL_DOWN = parameters["look_one_level_down"]
    MODEL_TYPE = parameters["model_type"]
    GPU = parameters["gpu"]

    output = io.load_train_test_data(
        TRAIN_DIR,
        TEST_DIR,
        image_filter=IMG_FILTER,
        mask_filter=MASK_FILTER,
        look_one_level_down=LOOK_ONE_LEVEL_DOWN,
    )
    images, labels, image_names, test_images, test_labels, image_names_test = output
    print("Data loaded")

    if config["running_time"]:
        model = models.CellposeModel(gpu=GPU, pretrained_model=config["results_dir"]+model_dir+"/"+running_time)
    else:
        model = models.CellposeModel(gpu=GPU, model_type=MODEL_TYPE)
    print(sum(p.numel() for p in model.net.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.net.parameters()))
    print("Model loaded")

    masks_pred, _, _ = model.eval(test_images, channels=[1, 3])
    print("Predictions done")

    _ = [imwrite(Path(config["results_dir"]+log_dir+"/"+
                str(Path(image_names_test[id]).stem)+'_pred'+parameters["extension"]), 
                masks_pred[id].astype(np.uint32)) for id in range(len(masks_pred))]

    example_cellpose(model=model,X=test_images[1],Y=test_labels[1],i=1,log_dir=log_dir, results_dir=config["results_dir"])

    print("Predictions saved")

if __name__ == "__main__":
    main()