import click
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from cellpose import io, models

from stardist.src.utils.config import read_json_config
from stardist.matching import matching_dataset


@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    parameters = config["parameters"]

    np.random.seed(int(parameters["seed"]))
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
        model = models.CellposeModel(pretrained_model=config["results_dir"]+"cellpose/"+
                                    config["running_time"]+"/"+config["running_time"],
                                    gpu=GPU)
    else:   
        model = models.CellposeModel(gpu=GPU, model_type=MODEL_TYPE)
    print("Model loaded")

    masks_pred, flows, _ = model.eval(test_images, channels=[1, 3])
    print("Predictions done")

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(test_labels, masks_pred, thresh=t, show_progress=False) for t in taus]

    for i in range(len(stats)):
        print(taus[i], "acc", round(stats[i].accuracy,3))
        print(taus[i], "iour", round(stats[i].mean_true_score,3))

    print("Scores Evaluated")

if __name__ == "__main__":
    main()