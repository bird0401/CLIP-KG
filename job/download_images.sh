#!/bin/sh -l

#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L jobenv=singularity
#PJM -g gk77
#PJM -j

module load singularity/3.7.3
singularity exec \
    --pwd /$HOME/CLIP-KG/src/ \
    --nv /$HOME/CLIP-KG/python_clip_v2_latest.sif \
    python download_images.py