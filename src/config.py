ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']

DATA_DIR = '/Users/cathywu/Dropbox/PhD/traffic-estimation-comparison/data'

# Replace with repository homes
REPOSITORIES = { \
        'synthetic': '/Users/cathywu/Dropbox/PhD/synthetic-traffic',
        'LS': '/Users/cathywu/Dropbox/PhD/traffic-estimation',
        'BI': '/Users/cathywu/Dropbox/PhD/traffic-bayesian',
        'CS': '/Users/cathywu/Dropbox/PhD/traffic-estimation',
        'wardrop': '/Users/cathywu/Dropbox/PhD/traffic-estimation-wardrop',
        'KSP': '/Users/cathywu/Dropbox/PhD/synthetic_traffic/grid_networks/YenKSP' }

import os
for (k,v) in REPOSITORIES.iteritems():
    # add to path after current dir
    os.sys.path.insert(1,v)
