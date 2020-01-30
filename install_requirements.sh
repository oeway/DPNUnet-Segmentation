#wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && bash Miniconda3-4.5.4-Linux-x86_64.sh -bfp /usr/local
#import sys
#sys.path.append('/usr/local/lib/python3.6/site-packages/')

# This is needed for windows - otherwise shapely will not install
conda config --add channels conda-forge
conda install shapely

# Pillow<7 is necessary to avoid error in torchvision:  ImportError: cannot import name 'PILLOW_VERSION'
conda install pillow=6.1

pip install tqdm

pip install descartes palettable geojson read-roi gputil namedlist
pip install lightgbm imgaug pandas imageio tensorboardX
conda install -y opencv tqdm scipy
conda install -y pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0 -c pytorch

# download pretrained weights
#wget -O Epoch_3259_fold0_best.pth https://kth.box.com/shared/static/q2gerflss0o2iv7y20535yy5rcwxr16u.pth
#mkdir -p weights-cell-cycle/dpn_softmax_f0
#cp Epoch_3259_fold0_best.pth weights-cell-cycle/dpn_softmax_f0/fold0_best.pth


