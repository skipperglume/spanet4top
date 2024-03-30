# From this: https://gitlab.mff.cuni.cz/mff/hpc/clusters/-/issues/20#note_61489
# Any user can request specific GPU RAM size using GRES vram, e.g. --gpus=1 --gres=vram:40G
# All GPUs have types, namely V100, A100, and L40
# V100 Tensor Core GPU from NVidisa - is the most advanced (16/32 GB ) VOLTA tesla? - worst card
# A100 Tensor Core GPU from NVidisa - (80 GB)  AMPERE - medium but fast slurm access
# L40 Tensor Core GPU from NVidisa - (48 GB) ADA LOVELACE - best card
# Can ask for a specific GPU using `gres` parameter, e.g. --gres=gpu:L40:2 allocates 2 L40 GPUs
# Don't forget to request GPU using --gpus=1

# Podman instead of docker 
# Charliecloud provides user-defined software stacks (UDSS) for HPC

# Need to try out ampere01 or ampere02 partitions
# sinfo for help

# To enter to a partition with gpu at future disposal: 
srun -p gpu-ffa --gpus=1 --time=2:00:00 --pty bash
    # Aiming for ampere GPU:
    srun -p gpu-ffa --gpus=1 --gres=gpu:A100:1,vram:80G --time=4:00:00 --pty bash
    srun -p gpu-ffa --gres=gpu:V100:1 --time=8:00:00 --pty bash
    srun -p gpu-ffa --gres=gpu:A100:1 --time=8:00:00 --pty bash
    srun -p gpu-ffa --gres=gpu:L40:1 --time=8:00:00  --pty bash 
    srun -p gpu-ffa --gres=gpu:L40:1 --time=8:00:00 --cpus-per-task=16 --pty bash 
    
    # Should land you in ampere02
# LAUNCH the container, e.g. 
ch-run -w -c /home/timoshyd --bind=/home/timoshyd -u 0 -g 0 ./spanet-img -- bash
# -w: write (should be dropped)
# -c Initial working directory in container 
--bind=/home/timoshyd/spanet4Top/:/home/timoshyd/spanet4Top/
# Tests:
    ch-run -w -c ./ --bind=/home/timoshyd/spanet4Top/:/home/timoshyd/spanet4Top/ -u 0 -g 0 ./spanet-img -- bash
    # launch the container, e.g. 
    ch-run -b ~:/mnt/0 ./spanet-img -- /bin/bash
# Check that nvidia is ok: (should print info about gpu state)
nvidia-smi
nvidia-smi | grep NVIDIA
nvidia-smi -a
# watch –n 1 -d nvidia-smi
# Verify and check conda existence by:
conda info
conda --version
# Start up conda and then environemnt 
conda init
conda activate spaconda
# Show the visible cude devices. A string of comma separated devices ids
echo $CUDA_VISIBLE_DEVICES
# conda create --name spaenv python=3.10
# conda activate spaenv
cd spanet4Top/SPANet/


# GPU support via tensorflow (it's not isntalled in conda environment)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# GPU support via torch
python -c "import torch; print('CUDA is set') if(torch.cuda.is_available()) else print('CUDA is not set'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); print(f'Current GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
# Need to understand parameters
# 0.45 it/s
python -m spanet.train -of options_files/full_hadronic_ttbar/example.json --time_limit 00:00:01:00 --gpus 0
# 2.87 it/s five times faster (for some reason changing --gpus to zero doesn't do anything)
# 14.27 it/s five times faster (for some reason changing --gpus to zero doesn't do anything)
python -m spanet.train -of options_files/full_hadronic_ttbar/example.json --time_limit 00:00:03:00 --gpus 4

# Increase batch_size from 4096 to  8192
# -b
# -cf checkpoint (what does it expect, which format?) -cf ./spanet_output/version_4
# To continue Last training
# -cf ./spanet_output/version_[NUMBER]/checkpoints/last.ckpt
# WITH
# -of ./spanet_output/version_[NUMBER]/options.json

python -m spanet.train -of options_files/full_hadronic_ttbar/full_training.json --gpus 1

python -m spanet.train -cf ./spanet_output/version_0/checkpoints/epoch=40-step=190363.ckpt -of options_files/full_hadronic_ttbar/full_training.json --gpus 4
python -m spanet.train  -of options_files/full_hadronic_ttbar/full_training.json --gpus 1

# python -m spanet.train  -of options_files/full_hadronic_ttbar/full_training.json --gpus 1
python -m spanet.train  -cf ./spanet_output/version_8/checkpoints/last.ckpt -of ./spanet_output/version_8/options.json  --gpus 1
# Testing: 
python -m spanet.test ./spanet_output/version_4 -tf data/full_hadronic_ttbar/testing.h5 --gpu
# Warning from volta01: (cou8ld be a problem)
# WARNING: infoROM is corrupted at gpu 0000:3B:00.0
SPANet/spanet_output/version_0/checkpoints/epoch=40-step=190363.ckpt
./spanet_output/version_0/checkpoints/epoch=40-step=190363.ckpt



# 4 top all-hadronic analysis training:
python -m spanet.train -of options_files/all_had_4top/example.json --gpus 1
python -m spanet.test ./spanet_output/version_0 -tf data/all_had_4top/four_top_SPANET_input_odd.h5 --gpu

// "learning_rate_cycles": 1,
// "num_jet_encoder_layers": 2,

python -m spanet.predict ./spanet_output/version_4 ./spanet_ttbar_testing_output.h5 -tf data/all_had_4top/output.h5 --gpu

