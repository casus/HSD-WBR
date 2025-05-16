import click
import numpy as np
from pathlib import Path
import time
from cellpose import io, models, train

from stardist.src.utils.config import read_json_config

io.logger_setup()

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))

def main(config_file_path):
    config = read_json_config(config_file_path)
    parameters = config["parameters"]

    np.random.seed(int(parameters["seed"]))

    running_time = time.strftime("%b-%d-%Y_%H-%M")
    print(running_time)

    # Creating log directory
    model_dir = 'cellpose' + '/' + f'{running_time}'
    Path(config["results_dir"]+model_dir).mkdir(parents=True,exist_ok=True)

    TRAIN_DIR =  parameters["train_dir"]
    VAL_DIR = parameters["val_dir"]
    IMG_FILTER = parameters["img_filter"]
    MASK_FILTER = parameters["mask_filter"]
    LOOK_ONE_LEVEL_DOWN = parameters["look_one_level_down"]
    PRETRAINED_MODEL = parameters["pretrained_model"]
    MODEL_TYPE = parameters["model_type"]
    GPU = parameters["gpu"]
    N_EPOCHS = parameters["n_epochs"]
    SAVE_PATH = parameters["save_path"]
    SAVE_EVERY = parameters["save_every"]

    images, labels, image_names, test_images, test_labels, image_names_test = (
        io.load_train_test_data(
            TRAIN_DIR,
            VAL_DIR,
            image_filter=IMG_FILTER,
            mask_filter=MASK_FILTER,
            look_one_level_down=LOOK_ONE_LEVEL_DOWN,
        )
    )

    if PRETRAINED_MODEL:
        model = models.CellposeModel(model_type=MODEL_TYPE, gpu=GPU)
    else:
        model = models.CellposeModel(pretrained_model=PRETRAINED_MODEL,
                                model_type=MODEL_TYPE, gpu=GPU)
    model_path = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        channels=[1, 3],
        test_data=test_images,
        test_labels=test_labels,
        n_epochs=N_EPOCHS,
        save_path=SAVE_PATH,
        save_every=SAVE_EVERY,
        model_name=running_time,
    )

if __name__ == "__main__":
    main()