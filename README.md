# HSD-WBR

This repository contains code for the experiments shown in the paper "Single-shot Star-convex Polygon-based Instance Segmentation for Spatially-correlated Biomedical Objects". [arXiv paper link](https://arxiv.org/html/2504.12078v1)

## Citation 
Please cite as follows:

De, T., Urbanski, A., Yakimovich, A.: Single-shot Star-convex Polygon-based Instance Segmentation for Spatially-correlated Biomedical Objects, http://arxiv.org/abs/2504.12078, (2025). https://doi.org/10.48550/arXiv.2504.12078.

```
@ARTICLE{De25,
  title        = "Single-shot star-convex polygon-based instance segmentation
                  for spatially-correlated biomedical objects",
  author       = "De, Trina and Urbanski, Adrian and Yakimovich, Artur",
  journal      = "CoRR",
  volume       = "abs/2504.12078",
  month        =  apr
  year         =  2025,
  primaryClass = "cs.CV",
}
```

## Abstract

Biomedical images often contain objects known to be spatially correlated or nested due to their inherent properties, leading to semantic relations. Examples include cell nuclei being nested within eukaryotic cells and colonies growing exclusively within their culture dishes. While these semantic relations bear key importance, detection tasks are often formulated independently, requiring multi-shot analysis pipelines. Importantly, spatial correlation could constitute a fundamental prior facilitating learning of more meaningful representations for tasks like instance segmentation. This knowledge has, thus far, not been utilised by the biomedical computer vision community. We argue that the instance segmentation of two or more categories of objects can be achieved in parallel. We achieve this via two architectures HydraStarDist (HSD) and the novel (HSD-WBR) based on the widely-used StarDist (SD), to take advantage of the star-convexity of our target objects. HSD and HSD-WBR are constructed to be capable of incorporating their interactions as constraints into account. HSD implicitly incorporates spatial correlation priors based on object interaction through a joint encoder. HSD-WBR further enforces the prior in a regularisation layer with the penalty we proposed named Within Boundary Regularisation Penalty (WBR). Both architectures achieve nested instance segmentation in a single shot. We demonstrate their competitiveness based on IoU_R and AP and superiority in a new, task-relevant criteria, Joint TP rate (JTPR) compared to their baseline SD and Cellpose. Our approach can be further modified to capture partial-inclusion/-exclusion in multi-object interactions in fluorescent or brightfield microscopy or digital imaging. Finally, our strategy suggests gains by making this learning single-shot and computationally efficient.


## Installation
Please set up the environment using `conda` or `pip`. We recommend creating a new environment for this project. Navigate to the root directory of this project and run:

```
conda create -f stardist_environment.yml
conda activate stardist2
pip install -e .
```

For some systems, the following line may be needed:
```
CC=gcc-<GCC VERSION> CXX=g++-<GCC VERSION> pip install -e .
```

For the cellpose environment, for some data-processing code, it may be needed for some scripts to have access to the custom code we have written into ```stardist/```, so you may need to run:

```
conda create -f cellpose_environment.yml
pip install -e .
conda activate cellpose
```

## Usage
Please use the scripts under ```examples/2D/models/stardist/```, ```examples/2D_hydra/models/stardist/``` and ```scripts/``` in combination with an appropriate config file from ```configs/``` to run the code. Please change the config with appropriate data, output and model weight paths.

Since HSD and HSD-WBR our branched architectures share plenty of common functionality with SD, a slight switch is needed to ensure the correct methods are being referred to. Please comment out one of the two lines below from ```stardist/models/__init__.py``` to use either a non-branched or branched(hydra) architecture.

```
from .model2d import Config2D, StarDist2D, StarDistData2D
from .model2d_hydra import Config2D, StarDist2D, StarDistData2D
```


## License
This repository is shared under the [MIT](https://choosealicense.com/licenses/mit/) license.

