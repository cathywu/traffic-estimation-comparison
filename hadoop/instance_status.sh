#!/bin/sh

# For use on local machine for checking the status of the various launched EC2 instances
# Checks if the mapper is still running and prints the output files which reside on the machine
# @author cathywu

for IP in 54.68.250.38 54.149.24.153 54.187.238.152
do
    echo "[[$IP]]"
    ssh -i ~/Dropbox/cathywu.pem ec2-user@$IP 'ps aux | grep python' | grep mapper
    ssh -i ~/Dropbox/cathywu.pem ec2-user@$IP 'ls -lah ~/output*'
done

