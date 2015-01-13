import os
import json

def chunk(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

if __name__ == "__main__":
    import config as c
    result_files = os.listdir(c.RESULT_DIR)
    scenario_files = os.listdir(c.SCENARIO_DIR)
    remainder = 'remainder/scenarios_remainder.txt'

    exclude = [ 'scenarios_test.txt', 'scenarios_temp.txt',
                'scenarios_small.txt', 'scenarios_all_sampled.txt' ]
    for f in exclude:
        if f in scenario_files:
            scenario_files.remove(f)

    scenarios = {}
    for file in scenario_files:
        filepath = "%s/%s" % (c.SCENARIO_DIR,file)
        if os.path.isdir(filepath):
            continue
        print file
        with open(filepath) as f:
            for line in f:
                d = json.loads(line)
                d['trial'] = 0
                if d['solver'] == 'BI':
                    continue
                key = frozenset(d.iteritems())
                if key not in scenarios:
                    scenarios[key] = 1
                else:
                    scenarios[key] += 1

    results = {}
    for file in result_files:
        filepath = "%s/%s" % (c.RESULT_DIR,file)
        if os.path.isdir(filepath):
            continue
        print file
        with open(filepath) as f:
            for line in f:
                if line[0] != '{':
                    continue

                try:
                    d = json.loads(line)
                except ValueError:
                    # ipdb.set_trace()
                    continue
                p = d['params']
                p['trial'] = 0
                key = frozenset(p.iteritems())
                if key not in results:
                    results[key] = 1
                else:
                    results[key] += 1

    num_files = 10
    size = len(scenarios)
    chunks = chunk(scenarios.items(), size/num_files)
    for i,chunkk in enumerate(chunks):
        with open('%s/%s.%s' % (c.SCENARIO_DIR,remainder,i), 'w') as f:
            for (k,v) in chunkk:
                if k not in results:
                    count = v
                else:
                    count = max(0,v-results[k])
                for i in xrange(count):
                    d = dict([x for x in k])
                    f.write("%s\n" % json.dumps(d))






