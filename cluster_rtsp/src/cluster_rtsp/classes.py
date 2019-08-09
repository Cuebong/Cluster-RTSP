import openravepy as orpy
import numpy as np


class InfoObj:
    def __init__(self):
        self.getconfig = 0.0
        self.configselection = 0.0
        self.clustering = 0.0
        self.globaltsp = 0.0
        self.localtsp = 0.0
        self.pathplanning = 0.0
        self.totalplanning = 0.0
        self.execution = 0.0
        self.N_clusters = 1


class SolverParameters(object):
    def __init__(self):
        # kinematics parameters
        self.iktype = orpy.IkParameterizationType.Transform6D
        self.qhome = np.deg2rad([0., -np.pi / 3., (2 * np.pi) / 3., 0., -np.pi / 3., 0.])
        self.standoff = 0.
        self.step_size = np.pi / 4.
        # Planning parameters
        self.try_swap = True
        self.planner = 'BiRRT'
        self.max_iters = 100
        self.max_ppiters = 30
