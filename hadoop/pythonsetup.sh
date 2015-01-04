#!/bin/bash
cd ~

sudo yum update -y 

sudo yum groupinstall -y development

sudo yum install -y zlib-dev openssl-devel sqlite-devel bzip2-devel

sudo yum install -y xz-libs

wget http://www.python.org/ftp/python/2.7.6/Python-2.7.6.tar.xz

xz -d Python-2.7.6.tar.xz

tar -xvf Python-2.7.6.tar

pushd Python-2.7.6
./configure
popd

sudo make && sudo make altinstall

export PATH="/usr/local/bin:$PATH"

wget --no-check-certificate https://pypi.python.org/packages/source/s/setuptools/setuptools-1.4.2.tar.gz

tar -xvf setuptools-1.4.2.tar.gz

pushd setuptools-1.4.2
sudo chmod -R 777 /usr/local/
python2.7 setup.py install
curl https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py | python2.7 -
popd

pip install virtualenv
