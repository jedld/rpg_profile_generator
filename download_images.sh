#!/bin/sh

mkdir -p generated_images
rsync -avz -e "ssh -p 51122" joseph@202.92.159.242:/home/joseph/workspace/rpg_profile_generator/generated_images generated_images
