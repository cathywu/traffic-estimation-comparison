#!/bin/bash

# For use on AWS instance, to update the various repositories (via git pull)
# @author cathywu

pushd ~/traffic-estimation
pushd comparison
git stash
popd

for repo in comparison synthetic_traffic BSC_NNLS bayesian
do
    pushd $repo
    echo "$repo"
    git pull
    popd
done

pushd comparison
python hadoop/maketrafficconfig.py `pwd`/..
popd

popd
