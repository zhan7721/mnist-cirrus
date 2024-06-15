# load python and pytorch
module load python/3.10.8-gpu
module load pytorch/1.13.1-gpu
# tensorflow/2.13.0-gpu

# add python to path (this makes sure we are using the right pip)
python_path=$(which python)
python_path="${python_path/python/""}" 
export PATH="$python_path:$PATH"

# set cache location and cd to work directory
USERNAME=$(whoami)
export HF_HOME="/work/tc062/tc062/${USERNAME}/.cache/hf"
export MPLCONFIGDIR="/work/tc062/tc062/${USERNAME}/.cache/matplotlib"
export WANDB_MODE="offline"

# create a default venv if it doesn't exist
export DEFAULT_VENV="/work/tc062/tc062/${USERNAME}/.venv/my_venv"
if [ ! -d "$DEFAULT_VENV" ]; then
    python -m venv --system-site-packages $DEFAULT_VENV
    extend-venv-activate $DEFAULT_VENV
fi

# activate the venv
export DEFAULT_VENV_ACTIVATE="${DEFAULT_VENV}/bin/activate"
source $DEFAULT_VENV_ACTIVATE
