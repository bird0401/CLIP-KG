#!/bin/sh
# $1: user_name, $2: image_name, $3: singularity_image_path

# sudo apt-get install squashfs-tools
singularity pull docker://$1/$2
singularity shell $3
