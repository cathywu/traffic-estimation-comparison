import ipdb

import logging
import os
import pickle
import random
from multiprocessing import Process

import config as c
from Scenario import Scenario
from scenario_utils import save, load

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
        self.tns.remove('test')
        self.scs.remove('test')
        self.solvers.remove('test')

        # Timing, tries, sampling parameters
        self.scan_interval = scan_interval
        self.sample_attempts = sample_attempts
        self.job_timeout = job_timeout

        self.test = test

        self.scan_done()
        self.load_long()

    def scan_done(self):
        self.done = {}
        scenario_files = os.listdir(self.scenario_dir)
        for sf in scenario_files:
            filename = "%s/%s" % (self.scenario_dir,sf)
            if os.path.isdir(filename):
                continue
            with open(filename) as f:
                s = pickle.load(f)
                key = (s.fname_tn, s.fname_sc, s.fname_solver)
                if key in self.done:
                    self.done[key] += 1
                else:
                    self.done[key] = 1

    def sample_new_scenario(self,attempts=1):
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

    def load_long(self):
        long = load(fname='%s/scenarios_long.pkl' % c.DATA_DIR)
        self.long = long if long is not None else {}

    def save_long(self):
        save(self.long, fname='%s/scenarios_long.pkl' % c.DATA_DIR)

    def run_experiment(self,jobs=100):
        myseed = 2347234328

        for i in xrange(jobs):
            fname_tn,fname_sc,fname_solver = \
                self.sample_new_scenario(attempts=self.sample_attempts)
            if fname_sc is None:
                break;
            key = (fname_tn,fname_sc,fname_solver)
            self.s = Scenario(fname_tn=fname_tn,fname_sc=fname_sc,
                         fname_solver=fname_solver,myseed=myseed,test=self.test)

            p = Process(target=self.run_job)
            p.start()
            p.join(self.job_timeout)
            if p.is_alive():
                # Terminate
                p.terminate()
                p.join()
                self.long[key] = 0
                self.save_long()
            else:
                self.done[key] = 1
                if self.test:
                    self.s.save(prefix='%s/test/Scenario')
                else:
                    self.s.save()

            # Occasionally update the set of finished tests
            if i > 0 and i % self.scan_interval == 0:
                self.scan_done()

if __name__ == "__main__":
    scan_interval = 100
    sample_attempts = 100
    job_timeout = 600
    njobs = 1000

    e = Experiment(c.TN_DIR,c.SC_DIR,c.SOLVER_DIR,c.SCENARIO_DIR_NEW,
                   scan_interval=scan_interval,
                   sample_attempts=sample_attempts,job_timeout=job_timeout)

    e.run_experiment(njobs)

