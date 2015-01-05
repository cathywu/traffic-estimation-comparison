#!/bin/bash

pushd ~
virtualenv trafficsetup
source ~/trafficsetup/bin/activate
mkdir traffic

pushd traffic
git clone https://github.com/cathywu/traffic-estimation-comparison
git clone https://github.com/cathywu/synthetic-traffic
git clone https://github.com/cathywu/traffic-estimation
git clone https://github.com/cathywu/traffic-estimation-bayesian
git clone https://github.com/jeromethai/traffic-estimation-wardrop
popd

pushd traffic/traffic-estimation-comparison
wget https://www.dropbox.com/s/pzp6ncb06ksmzsv/maketrafficconfig.py
python maketrafficconfig.py ..
popd

sudo yum install -y lapack lapack-devel blas blas-devel
sudo yum install -y libpng-devel freetype-devel

pip install numpy 
pip install scipy
pip install ipython networkx cvxopt scikit-learn matplotlib ipdb
sudo yum install ncurses-devel
pip install readline

# Setup specific to traffic-estimation
pushd traffic/traffic-estimation/python/c_extensions
python setup.py build_ext --inplace
popd

# Setup specific to traffic-estimation-bayesian
# sudo yum install python27-devel
pushd ~
git clone https://github.com/pymc-devs/pymc.git
pushd pymc
git checkout 3bf4ad3285f658d02a6b4297160b45354666fe46
python setup.py install
popd
popd

deactivate

