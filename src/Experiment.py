import ipdb

import logging
import os
import cPickle as pickle
import random
from multiprocessing import Process
import json

import config as c
from Scenario import Scenario
from scenario_utils import save, load, args_from_SC, args_from_solver, args_from_TN

class Experiment:

    def __init__(self, tn_dir, sc_dir, solver_dir, scenario_dir,
                 scan_interval=100, sample_attempts=100, job_timeout=600,
                 test=False):
        self.tn_dir = tn_dir
        self.sc_dir = sc_dir
        self.solver_dir = solver_dir
        self.scenario_dir = scenario_dir

        self.tns = os.listdir(self.tn_dir)
        self.scs = os.listdir(self.sc_dir)
        self.solvers = os.listdir(self.solver_dir)
        for exclude in ['test','temp','all']:
            for l in [self.tns, self.scs, self.solvers]:
                if exclude in l:
                    l.remove(exclude)

        # Timing, tries, sampling parameters
        self.scan_interval = scan_interval
        self.sample_attempts = sample_attempts
        self.job_timeout = job_timeout

        self.test = test

        self.scan_done()
        self.load_long()

    def scan_done(self):
        self.done = {}
        logging.info('Scan start')
        scenario_files = os.listdir(self.scenario_dir)
        for sf in scenario_files:
            filename = "%s/%s" % (self.scenario_dir,sf)
            if os.path.isdir(filename):
                continue
            with open(filename) as f:
                try:
                    s = pickle.load(f)
                    key = (s.fname_tn, s.fname_sc, s.fname_solver)
                    if key in self.done:
                        self.done[key] += 1
                    else:
                        self.done[key] = 1
                except EOFError:
                    print 'Could not load, please delete: %s' % filename
        logging.info('Scan done')


    def sample_new_scenario(self,attempts=1):
        logging.info('Sampling new scenario')
        for i in xrange(attempts):
            fname_tn = random.choice(self.tns)
            fname_sc = random.choice(self.scs)
            fname_solver = random.choice(self.solvers)
            key = (fname_tn,fname_sc,fname_solver)

            if key not in self.done and key not in self.long:
                return fname_tn, fname_sc, fname_solver
        logging.error('Error: max attempts reached when sampling for new scenario')
        return None,None,None

    def run_job(self):
        self.s.run()
        if self.test:
            self.s.save(prefix='%s/test/Scenario')
        else:
            self.s.save()

    def load_long(self):
        long = load(fname='%s/scenarios_long.pkl' % c.DATA_DIR)
        self.long = long if long is not None else {}
        logging.info('Loading of "long" done')

    def save_long(self):
        save(self.long, fname='%s/scenarios_long.pkl' % c.DATA_DIR)

    @staticmethod
    def get_available(fnames, dir, args_fn):
        args_list = []
        for fname in fnames:
            fpath = '%s/%s' % (dir,fname)
            with open(fpath) as f:
                obj = pickle.load(f)
            args = args_fn(obj)
            args_list.append((args, fname))
        return args_list

    @staticmethod
    def find(params, args_list, keys):
        param_keys = {}
        for key in keys:
            if key in params:
                param_keys[key] = params[key]

        for (args,fname) in args_list:
            if frozenset(args.iteritems()) == frozenset(param_keys.iteritems()):
                return fname
        return None

    def run_experiment_from_file(self,fname):
        myseed = 2347234328
        TNs = Experiment.get_available(self.tns, c.TN_DIR, args_from_TN)
        SCs = Experiment.get_available(self.scs, c.SC_DIR, args_from_SC)
        solvers = Experiment.get_available(self.solvers, c.SOLVER_DIR,
                                           args_from_solver)

        with open(fname) as todo:
            lines = todo.readlines()
            random.shuffle(lines)
            for (i,line) in enumerate(lines):
                params = json.loads(line)
                fname_tn = Experiment.find(params, TNs,
                                           ['model', 'nrow', 'ncol', 'sparse',
                                            'nodroutes'])
                fname_sc = Experiment.find(params, SCs,
                                           ['NS', 'NL', 'NB', 'NLP'])
                fname_solver = Experiment.find(params, solvers,
                                               ['solver', 'method', 'init',
                                                'noise', 'eq', 'all_links',
                                                'use_L', 'use_OD', 'use_CP',
                                                'use_LP', 'sparse_BI'])
                key = (fname_tn,fname_sc,fname_solver)

                if fname_tn is None or fname_sc is None or fname_solver is None:
                    print "Scenario configuration not found: %s" % params
                    continue

                if key in self.done:
                    print "Already done: %s" % params
                    continue

                print 'Found job to do: %s' % params
                args = argparse.Namespace()
                args.__dict__ = params
                self.s = Scenario(fname_tn=fname_tn, fname_sc=fname_sc,
                                  fname_solver=fname_solver, myseed=myseed,
                                  test=self.test, args=args)

                logging.info('Running job')
                p = Process(target=self.run_job)
                p.start()
                p.join(self.job_timeout)
                if p.is_alive():
                    logging.error("Error (timeout): terminating job %s" % repr(key))
                    # Terminate
                    p.terminate()
                    p.join()
                    self.long[key] = 0
                    self.save_long()
                else:
                    self.done[key] = 1

                # Occasionally update the set of finished tests
                if i > 0 and i % self.scan_interval == 0:
                    self.scan_done()

    def run_experiment(self,jobs=100):
        myseed = 2347234328

        for i in xrange(jobs):
            fname_tn,fname_sc,fname_solver = \
                self.sample_new_scenario(attempts=self.sample_attempts)
            if fname_sc is None:
                break;
            key = (fname_tn,fname_sc,fname_solver)
            self.s = Scenario(fname_tn=fname_tn, fname_sc=fname_sc,
                              fname_solver=fname_solver, myseed=myseed,
                              test=self.test)

            logging.info('Running job')
            p = Process(target=self.run_job)
            p.start()
            p.join(self.job_timeout)
            if p.is_alive():
                logging.error("Error (timeout): terminating job %s" % repr(key))
                # Terminate
                p.terminate()
                p.join()
                self.long[key] = 0
                self.save_long()
            else:
                self.done[key] = 1

            # Occasionally update the set of finished tests
            if i > 0 and i % self.scan_interval == 0:
                self.scan_done()

if __name__ == "__main__":
    scan_interval = 100
    sample_attempts = 300
    job_timeout = 9000
    njobs = 1000

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', dest='fname', type=str, default=None,
                        help='File with scenario configurations')
    args = parser.parse_args()
    fname = args.fname

    e = Experiment(c.TN_DIR,c.SC_DIR,c.SOLVER_DIR,c.SCENARIO_DIR_NEW,
                   scan_interval=scan_interval,
                   sample_attempts=sample_attempts,job_timeout=job_timeout)

    if fname is not None:
        e.run_experiment_from_file(fname)
    else:
        e.run_experiment(njobs)
