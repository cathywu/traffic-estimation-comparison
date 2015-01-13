#!/bin/bash

# For use on AWS instance, to update the various repositories (via git pull)
# @author cathywu

pushd ~/traffic
pushd traffic-estimation-comparison
git stash
popd

for repo in traffic-estimation-comparison synthetic-traffic traffic-estimation traffic-estimation-bayesian traffic-estimation-wardrop
do
    pushd $repo
    echo "$repo"
    git pull
    popd
done

pushd traffic-estimation-comparison
python maketrafficconfig.py `pwd`/..
popd

popd
