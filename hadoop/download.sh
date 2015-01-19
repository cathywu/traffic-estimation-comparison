#!/bin/sh

while true
do
    echo
    for IP in 54.149.192.119 54.148.67.33 54.149.199.26 54.186.169.185 54.148.95.238 54.186.37.27 54.69.0.22 54.68.250.38 54.187.247.247 54.187.253.113 54.149.24.153 54.187.238.152 54.186.86.201 54.186.1.23 54.187.169.32 54.191.137.184
    do
        echo "[[$IP]]"
        scp -i ~/Dropbox/cathywu.pem ec2-user@$IP:output_\*.txt\* /Users/cathywu/Dropbox/PhD/traffic-estimation-comparison/output
    done
    sleep 600
done

