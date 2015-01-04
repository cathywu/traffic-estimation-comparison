#!/usr/bin/env python
import sys
import numpy as np
import json
import os
import numpy, scipy
import base64

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def setup():
    # path = '/home/hadoop/traffic-estimation-comparison/src'
    path = '/Users/cathywu/Dropbox/PhD/traffic-estimation-comparison/src'
    os.sys.path.insert(1,path)

def scenario(params):
    import sys
    import traceback
    try:
        return Scenario.scenario(params)
    except:
        print traceback.format_exc()
        traceback.print_exc()
        # print "Unexpected error:", sys.exc_info()[0]
    # return json.dumps(params) # replace with your function

# input comes from STDIN (standard input)
setup()
import Scenario
for line in sys.stdin:
    line = line.strip()
    if len(line) == 0:
        continue
    params = json.loads(line)
    output = scenario(params)
    output['params'] = params
    print json.dumps(output, cls=NumpyAwareJSONEncoder)
    # print json.dumps(base64.urlsafe_b64encode(scenario(params)))
    # print base64.urlsafe_b64decode(json.loads(json.dumps(base64.urlsafe_b64encode(content))).encode('ascii'))
