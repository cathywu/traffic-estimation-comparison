#!/bin/bash

# For use on AWS instance, to run a particular scenarios file
# USAGE: . run_scenarios all 1
# @author cathywu

rm trafficupdate.sh
wget https://s3-us-west-2.amazonaws.com/comparisonpaper/hadoop-helpers/trafficupdate.sh
. trafficupdate.sh

rm scenarios_$1.txt
wget https://s3-us-west-2.amazonaws.com/comparisonpaper/hadoop-helpers/input/scenarios_$1.txt
python ~/traffic/traffic-estimation-comparison/hadoop/mapper.py < scenarios_$1.txt > output_$1.txt.$2
