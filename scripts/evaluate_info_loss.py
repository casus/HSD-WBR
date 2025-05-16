import ast
import tensorflow as tf
from albumentations.core.serialization import load as load_albumentations_transform
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
import glob
import random
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union

from stardist.src.data.image_mask_datasets import ImageMaskDataset
from stardist.src.models.metrics import DiceCoeff
from stardist.src.utils.config import read_json_config
from stardist.src.utils.data import split_into_patches

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
        "hamming": pil_image.HAMMING,
        "box": pil_image.BOX,
        "lanczos": pil_image.LANCZOS,
    }

def evaluate_model(dataset_low_res, dataset, target_height, target_width):
    dice_coeff = DiceCoeff()

    for idx in range(len(list(zip(dataset_low_res,dataset)))):
        _, mask_low_res = dataset_low_res[idx]
        _, mask = dataset[idx]

        mask_low_res, mask = np.squeeze(mask_low_res), np.squeeze(mask)
        width, height = pil_image.fromarray(mask).size
        temp = pil_image.fromarray(mask_low_res).resize((width, height))
        mask_upscaled = np.array(temp)


        mask_patches = split_into_patches(mask, target_height, target_width)
        mask_upscaled_patches = split_into_patches(mask_upscaled, target_height, target_width)

        dice_coeff.update_state(mask_upscaled_patches[0], mask_patches[0])

    return {"avg_dice_coeff": dice_coeff.result().numpy()}

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as exc:
            raise click.BadParameter(value) from exc


@click.command()
@click.option("--config_file_path", type=click.Path(exists=True))

@click.option(
    "--iter", default=5, type=int,
    help="Number of times to iterate experiment for comparing info loss", show_default=True
    )


def main(config_file_path, iter):
    config = read_json_config(config_file_path)

    if config["results_dir"] is not None:
        results_dir = Path(config["results_dir"])
    else:
        results_dir = Path(__file__).parents[1]

    # TF dimension ordering in this code
    K.set_image_data_format("channels_last")

    parameters = config["parameters"]

    tf.random.set_seed(parameters["seed"])

    running_time = time.strftime("%b-%d-%Y_%H-%M")
    model_dir = results_dir / "model"

    log_dir = model_dir / "logs"/ f"{running_time}"
    log_dir.mkdir(parents=True, exist_ok=True)


    transform = load_albumentations_transform(config["test"]["transform_filepath"])

    fold_res_l = [1.0/(2**i) for i in range(int(parameters['n'])+1)]

    info_loss_iter_ls = find_info_loss2(fold_res_l,iter, transform, config, parameters)

    info_loss_iter_ls = np.array(info_loss_iter_ls).flatten()
    print(len(info_loss_iter_ls))
    print(len(np.array([[fold_res_l]*iter]).flatten()))

    info_loss_df = pd.DataFrame({
        "dice_coeff": info_loss_iter_ls,
        "fold": np.array([[fold_res_l]*iter]).flatten()
    })

    print(info_loss_df)

    # boxplot_info_loss(info_loss_df, save_path = log_dir / "graph_info_loss.svg")
    # barplot_info_loss(info_loss_df, save_path = log_dir / "graph_info_loss_plaque.svg")

def find_info_loss2(fold_res_l, it, transform, config, parameters):

    info_loss_iter_ls = []
    i = 0

    image_mask_dataset_test = ImageMaskDataset(
        Path(config["test"]["img_dir"]),
        Path(config["test"]["mask_dir"]),
        transform=transform,
        num_classes=parameters["num_classes"],
        batch_size=1,
        shuffle=False,
    )
    p_ls = ['./configs/transforms/1_fold_res/test.json',
        './configs/transforms/0_5_fold_res/test.json',
        './configs/transforms/0_25_fold_res/test.json',
        './configs/transforms/0_125_fold_res/test.json',
        './configs/transforms/0_0625_fold_res/test.json',
        './configs/transforms/0_03125_fold_res/test.json']

    while i < it:
        info_loss_over_res_ls = []
        for j in range(len(list(zip(fold_res_l,p_ls)))):
            transform_low_res_path = p_ls[j]
            print(transform_low_res_path)
            transform_low_res = load_albumentations_transform(transform_low_res_path)
            image_mask_dataset_test_low_res = ImageMaskDataset(
                Path(config["test"]["img_dir"]),
                Path(config["test"]["mask_dir"]),
                transform=transform_low_res,
                num_classes=parameters["num_classes"],
                batch_size=1,
                shuffle=False,
            )

            metrics = evaluate_model(
                image_mask_dataset_test_low_res,
                image_mask_dataset_test,
                parameters["target_height"],
                parameters["target_width"],
            )
            info_loss_over_res_ls.append(metrics['avg_dice_coeff'])

        info_loss_over_res_ls = np.array(info_loss_over_res_ls)
        info_loss_iter_ls.append(info_loss_over_res_ls)

        i+=1

    return info_loss_iter_ls

def boxplot_info_loss(info_loss_df, save_path):
    sns.boxplot(
        data=info_loss_df,
        x="fold",
        y="dice_coeff",
        notch=False,
        showcaps=True,
        flierprops={"marker": "x"},
        boxprops={"facecolor": (.4, .6, .8, .5)},
        medianprops={"color": "coral"}
    ).set(xlabel="Fold Resolution", ylabel="Dice Coefficient")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

def barplot_info_loss(info_loss_df, save_path):
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    sns.barplot(
        x="fold",
        y="dice_coeff",
        data=info_loss_df,
        errorbar="sd",
        capsize=.2,
        color="black",edgecolor="black"
    ).set(xlabel="Fold resolution", ylabel="Dice coefficient")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
