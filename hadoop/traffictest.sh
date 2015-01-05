#!/bin/bash

source ~/trafficsetup/bin/activate

pushd traffic/traffic-estimation-comparison
python2.7 src/Scenario.py --log INFO --output --NLP 33 --NL 33 --nrow 3 --solver BI --NB 160 --ncol 2 --nodroutes 15 --model P --NS 0 --method LBFGS --sparse

python2.7 src/generate_scenarios.py

pushd hadoop
python2.7 mapper.py < /home/ec2-user/traffic/traffic-estimation-comparison/scenarios_all_sampled.txt > output.txt

popd
popd

# deactivate
