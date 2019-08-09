import numpy as np
import time
import openravepy as orpy
import raveutils as ru


def compute_robot_configurations(env, robot, targets, params, qstart=None):
    # Compute the IK solutions
    starttime = time.time()
    configurations = None
    configuration_ID = 1
    pos_check = np.array([0.56, -0.23, 0.305])

    if qstart is not None:
        configurations.append([qstart])
    for i, ray in enumerate(targets):
        newpos = ray.pos() - params.standoff * ray.dir()
        if np.sqrt(np.sum((newpos - pos_check) ** 2)) < 0.0001:
            print(configuration_ID)
        newray = orpy.Ray(newpos, ray.dir())
        solutions = ru.kinematics.find_ik_solutions(robot, newray, params.iktype, collision_free=True, freeinc=params.step_size)
        if np.size(solutions) > 0:
            ID = np.array([[configuration_ID]] * len(solutions))
            configuration_ID += 1
            point = np.array([ray.pos()] * len(solutions))
            solutions = np.hstack((solutions, point, ID, np.array([[i]] * len(solutions))))
            if configurations is None:
                configurations = solutions
            else:
                configurations = np.vstack((configurations, solutions))

    cpu_time = time.time() - starttime
    return configurations, cpu_time


def compute_execution_time(trajectories):
    execution_time = 0
    for traj in trajectories:
        if traj is None:
            print('Empty trajectory found - ignoring...')
            continue
        execution_time += traj.GetDuration()
    return execution_time
