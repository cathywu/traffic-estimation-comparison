#!/bin/bash

pushd traffic/traffic-estimation-comparison
python2.7 src/generate_scenarios.py

pushd hadoop
python2.7 mapper.py < /home/ec2-user/traffic/traffic-estimation-comparison/scenarios_all_sampled.txt > output.txt

popd
popd
