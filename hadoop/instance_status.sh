#!/bin/sh

# For use on local machine for checking the status of the various launched EC2 instances
# Checks if the mapper is still running and prints the output files which reside on the machine
# @author cathywu

for IP in 54.149.192.119 54.148.67.33 54.149.199.26 54.186.169.185 54.186.241.246 54.148.95.238 54.68.168.159 54.186.37.27 54.69.0.22 54.68.250.38 54.187.247.247 54.187.253.113 54.149.24.153 54.187.238.152 54.186.86.201 54.186.1.23
do
    echo "[[$IP]]"
    ssh -i ~/Dropbox/cathywu.pem ec2-user@$IP 'ps aux | grep python' | grep mapper
    ssh -i ~/Dropbox/cathywu.pem ec2-user@$IP 'ls -lah ~/output*'
done

