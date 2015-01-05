#!/bin/bash

source ~/trafficsetup/bin/activate

# test a particular scenario
pushd traffic/traffic-estimation-comparison
python2.7 src/Scenario.py --log INFO --output --NLP 33 --NL 33 --nrow 3 --solver BI --NB 160 --ncol 2 --nodroutes 15 --model P --NS 0 --method LBFGS --sparse

# generate scenarios
python src/generate_scenarios.py

# test mapper on a scenarios file
python hadoop/mapper.py < /home/ec2-user/traffic/traffic-estimation-comparison/scenarios_test.txt > output.txt
popd

deactivate
