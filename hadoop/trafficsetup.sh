#!/bin/bash

cd ~
mkdir traffic
pushd traffic

git clone https://github.com/cathywu/traffic-estimation-comparison
git clone https://github.com/cathywu/synthetic-traffic
git clone https://github.com/cathywu/traffic-estimation
git clone https://github.com/cathywu/traffic-estimation-bayesian
git clone https://github.com/jeromethai/traffic-estimation-wardrop
wget https://www.dropbox.com/s/pzp6ncb06ksmzsv/maketrafficconfig.py
python2.7 maketrafficconfig.py .

sudo yum install -y lapack lapack-devel blas blas-devel
sudo yum install -y libpng-devel freetype-devel

pip install numpy 
pip install scipy
pip install ipython networkx cvxopt scikit-learn pymc matplotlib

pushd traffic-estimation/python/c_extensions
python2.7 setup.py build_ext --inplace

popd
popd
