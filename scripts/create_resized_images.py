import json
from pathlib import Path
from typing import Union

import click
from PIL import Image

from stardist.src.utils.config import read_json_config

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)

    current_dir	= config['project_result_dir']
    if current_dir:
        current_dir = Path(current_dir)
    else:
        current_dir = Path(__file__).parent

    print(current_dir)

    source_files = Path(config['conversion']['source_dir']) \
                    .glob('*.'+config['conversion']['source_type'])
    target_dir = Path(config['conversion']['target_dir'])

    n = int(config['conversion']['n'])
    for source in source_files:
        image = Image.open(source)

        for i in range(n+1):
            width, height = image.size
            temp = image.resize((width//(2**i), height//(2**i)))
            # print(temp.size)
            # temp.save(target_dir / (source.stem + '_' + str(i) + '.' + config['conversion']['target_type']))


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
