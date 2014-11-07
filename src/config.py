ACCEPTED_LOG_LEVELS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'WARN']

DATA_DIR = '/Users/cathywu/Dropbox/PhD/traffic-estimation-comparison/data'

# Replace with repository homes
REPOSITORIES = { \
        'wardrop': '/Users/cathywu/Dropbox/PhD/traffic-estimation-wardrop',
        'synthetic': '/Users/cathywu/Dropbox/PhD/synthetic_traffic',
        'LS': '/Users/cathywu/Dropbox/PhD/traffic-estimation',
        'BI': '/Users/cathywu/Dropbox/PhD/traffic-bayesian',
        'CS': '/Users/cathywu/Dropbox/PhD/traffic-estimation',
        'KSP': '/Users/cathywu/Dropbox/PhD/synthetic_traffic/grid_networks/YenKSP' }

import os
for (k,v) in REPOSITORIES.iteritems():
    os.sys.path.append(v)
