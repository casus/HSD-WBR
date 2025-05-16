#!/bin/sh

# change "seed" in config between runs
python examples/2D/train_stardist.py configs/sample_train_stardist_config.json > examples/2D/models/stardist/hela_cytoplasm_run1.out

python examples/2D/train_stardist.py configs/sample_train_stardist_config.json > examples/2D/models/stardist/hela_cytoplasm_run2.out

python examples/2D/train_stardist.py configs/sample_train_stardist_config.json > examples/2D/models/stardist/hela_cytoplasm_run3.out



python examples/2D/train_hydranet_stardist.py configs/sample_train_hydranet_stardist_config.json > examples/2D_hydra/models/stardist/hela_hydra_run1.out

python examples/2D/train_hydranet_stardist.py configs/sample_train_hydranet_stardist_config.json > examples/2D_hydra/models/stardist/hela_hydra_run2.out

python examples/2D/train_hydranet_stardist.py configs/sample_train_hydranet_stardist_config.json > examples/2D_hydra/models/stardist/hela_hydra_run3.out
