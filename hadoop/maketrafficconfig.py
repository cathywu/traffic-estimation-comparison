import sys

if len(sys.argv) != 2:
	print "the number of arguments is", len(sys.argv) - 1, " when expected 1." 
	raise Exception

base = sys.argv[1].strip()

config = '''ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']

DATA_DIR = '{0}/traffic-estimation-comparison/data'

# Replace with repository homes
REPOSITORIES = { 
        'synthetic': '{0}/synthetic-traffic',
        'LS': '{0}/traffic-estimation',
        'BI': '{0}/traffic-estimation-bayesian',
        'CS': '{0}/traffic-estimation',
        'wardrop': '{0}/traffic-estimation-wardrop',
        'KSP': '{0}/synthetic_traffic/grid_networks/YenKSP' }

import os
for (k,v) in REPOSITORIES.iteritems():
    # add to path after current dir
    os.sys.path.insert(1,v)
'''
f = open(base + "/traffic-estimation-comparison/src/config.py", 'w')
f.write(config.replace("{0}", base))
