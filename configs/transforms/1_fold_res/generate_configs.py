
from pathlib import Path

import click
from albumentations import Affine, Compose, HorizontalFlip, RandomCrop, ToFloat, VerticalFlip

from stardist.src.utils.config import transform_to_json


@click.command()
@click.argument(
    'transform_configs_dir',
    type=click.Path(exists=True),
    default=Path(__file__).parent.resolve(), # directory of this file
    )
def main(transform_configs_dir):
    transform_configs_dir = Path(transform_configs_dir)

    train_transform = Compose([
        ToFloat(max_value=255),
        VerticalFlip(),
        HorizontalFlip(),
        Affine(scale=1.0, fit_output=True, keep_ratio=True, p=1),
        RandomCrop(width=256, height=256),
    ])
    transform_to_json(train_transform, transform_configs_dir / 'train.json')

    val_transform = Compose([
        ToFloat(max_value=255),
        Affine(scale=1.0, fit_output=True, keep_ratio=True, p=1),
        RandomCrop(width=256, height=256),
    ])
    transform_to_json(val_transform, transform_configs_dir / 'val.json')

    # As we split the test images into all the patches, crop should not be used in test augmentations
    test_transform = Compose([
        ToFloat(max_value=255),
        Affine(scale=1.0, fit_output=True, keep_ratio=True, p=1),
    ])
    transform_to_json(test_transform, transform_configs_dir / 'test.json')

    test_transform1 = Compose([
        ToFloat(max_value=255),
        Affine(scale=0.5, fit_output=True, keep_ratio=True, p=1),
    ])
    transform_to_json(test_transform1, transform_configs_dir / 'test1.json')

    test_transform2 = Compose([
        ToFloat(max_value=255),
        Affine(scale=0.25, fit_output=True, keep_ratio=True, p=1),
    ])
    transform_to_json(test_transform2, transform_configs_dir / 'test2.json')

    test_transform3 = Compose([
        ToFloat(max_value=255),
        Affine(scale=0.125, fit_output=True, keep_ratio=True, p=1),
    ])
    transform_to_json(test_transform3, transform_configs_dir / 'test3.json')

    test_transform4 = Compose([
        ToFloat(max_value=255),
        Affine(scale=0.0625, fit_output=True, keep_ratio=True, p=1),
    ])
    transform_to_json(test_transform4, transform_configs_dir / 'test4.json')

    test_transform5 = Compose([
        ToFloat(max_value=255),
        Affine(scale=0.03125, fit_output=True, keep_ratio=True, p=1),
    ])
    transform_to_json(test_transform5, transform_configs_dir / 'test5.json')

if __name__ == '__main__':
    main() # pylint: disable=no-value-for-parameter
