import time

import config as c
from SolverBI import SolverBI
from SolverLSQR import SolverLSQR
from SolverLS import SolverLS
from SolverCS import SolverCS

from scenario_utils import save

if __name__ == "__main__":
    solvers = []
    # PRIORITY 1
    # ---------------------------------------------------------------
    solvers.append(SolverLSQR(damp=0))
    solvers.append(SolverCS(method='cvx_random_sampling_L1_30_replace'))

    solvers.append(SolverLS(init=True,method='BB'))
    solvers.append(SolverLS(init=False,method='BB'))

    solvers.append(SolverBI(sparse=0.3))

    # PRIORITY 2
    # ---------------------------------------------------------------
    # solvers.append(SolverBI(sparse=0))
    # solvers.append(SolverBI(sparse=1))
    # solvers.append(SolverBI(sparse=2))

    # solvers.append(SolverLS(init=True,method='LBFGS'))
    # solvers.append(SolverLS(init=False,method='LBFGS'))
    # solvers.append(SolverLS(init=True,method='DORE'))
    # solvers.append(SolverLS(init=False,method='DORE'))

    # solvers.append(SolverCS(method='cvx_oracle'))
    # solvers.append(SolverLSQR(damp=1))

    for s in solvers:
        save(s, prefix="%s/Solver" % c.SOLVER_DIR)
