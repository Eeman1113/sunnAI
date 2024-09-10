# sunnAI
sunnAI an endless song webpage
```
# must have miniconda installed

# installation for musicgen
conda create -n audiocraft python=3.9
conda activate audiocraft
brew install ffmpeg (for mac)
winget install ffmpeg (for windows)
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
python -m pip install -r requirements.txt
python -m demos.musicgen_app --share
# first time you run inference, it'll download the model

# installation for audiogen
# NOTE: after already installing music gen
# NOTE: after PR185 gets merged, you won't have to execute the next two commands
git fetch origin pull/185/head:PR185
git checkout PR185
python -m demos.audiogen_app --share
conda install -c pytorch pytorch
# first time you run inference, it'll download the model
```
