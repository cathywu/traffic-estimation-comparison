#!/bin/sh

while true
do
    echo
    for IP in 54.68.250.38 54.149.24.153 54.187.238.152

    do
        echo "[[$IP]]"
        scp -i ~/Dropbox/cathywu.pem ec2-user@$IP:output_\*.txt\* /Users/cathywu/Dropbox/PhD/traffic-estimation-comparison/output
    done
    sleep 600
done

