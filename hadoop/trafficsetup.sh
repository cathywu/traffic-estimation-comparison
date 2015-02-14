#!/bin/bash

pushd ~
virtualenv trafficsetup
popd

source ~/trafficsetup/bin/activate

pushd ~
mkdir traffic-estimation
pushd traffic-estimation
touch __init__.py
git clone https://github.com/cathywu/traffic-estimation-comparison comparison
git clone https://github.com/cathywu/synthetic-traffic synthetic_traffic
git clone https://github.com/cathywu/traffic-estimation BSC_NNLS
git clone https://github.com/cathywu/traffic-estimation-bayesian bayesian
popd

pushd traffic-estimation/comparison
python hadoop/maketrafficconfig.py `pwd`/..
mkdir data
mkdir data/sensor_configurations
mkdir data/solvers
mkdir data/networks
mkdir data/scenarios
mkdir data/solvers/test
mkdir data/sensor_configurations/test
mkdir data/networks/test
mkdir data/scenarios/test
popd

sudo yum install -y lapack lapack-devel blas blas-devel
sudo yum install -y libpng-devel freetype-devel

pip install numpy 
pip install scipy
pip install ipython networkx cvxopt scikit-learn matplotlib ipdb
yes | sudo yum install ncurses-devel
pip install readline

# Setup specific to traffic-estimation
pushd traffic-estimation/BSC_NNLS/python/c_extensions
python setup.py build_ext --inplace
popd

popd

# Setup specific to traffic-estimation-bayesian
# sudo yum install python27-devel
pushd ~
git clone https://github.com/pymc-devs/pymc.git
pushd pymc
git checkout 3bf4ad3285f658d02a6b4297160b45354666fe46
python setup.py install
popd

# deactivate

