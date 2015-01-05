#!/bin/bash
cd ~

# Update EC2 default instance
sudo yum update -y 

sudo yum groupinstall -y development
sudo yum install -y zlib-dev openssl-devel sqlite-devel bzip2-devel
sudo yum install -y xz-libs

# Setup Python 2.7 (default installed is 2.6)
wget http://www.python.org/ftp/python/2.7.6/Python-2.7.6.tar.xz
xz -d Python-2.7.6.tar.xz
tar -xvf Python-2.7.6.tar

pushd Python-2.7.6
./configure --prefix=/usr/local --enable-unicode=ucs4 --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib"
sudo make && sudo make altinstall
popd

export PATH="/usr/local/bin:$PATH"

# Setup setuptools and pip
# wget --no-check-certificate https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
# python2.7 ez_setup.py
# easy_install-2.7 pip

wget --no-check-certificate https://bootstrap.pypa.io/ez_setup.py
python2.7 ez_setup.py --insecure

wget --no-check-certificate https://pypi.python.org/packages/source/s/setuptools/setuptools-11.1.tar.gz
tar -xvf setuptools-11.1.tar.gz

pushd setuptools-11.1
sudo chmod -R 777 /usr/local/
python2.7 setup.py install
popd

# install pip
easy_install-2.7 pip
# curl https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py | python -

# Setup virtualenv (though currently not used)
pip install virtualenv
