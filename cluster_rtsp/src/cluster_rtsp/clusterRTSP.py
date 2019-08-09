import numpy as np
from timeit import default_timer as timer
import time
import csv
import openravepy as orpy
import raveutils as ru
import environment as environments
import XMeans
import kinematics
import tsp
import configselect as cselect
import classes
import rospkg


def play_trajectory(env, robot, traj, play_speed=1.0, rate=100.):
    indices = robot.GetActiveManipulator().GetArmIndices()
    spec = traj.GetConfigurationSpecification()
    starttime = time.time()
    while time.time() - starttime < traj.GetDuration() / play_speed:
        current_time = (time.time() - starttime) * play_speed
        with env:  # have to lock environment since accessing robot
            trajdata = traj.Sample(current_time)
            qrobot = spec.ExtractJointValues(trajdata, robot, indices, 0)
            robot.SetActiveDOFValues(qrobot)
        time.sleep(1. / rate)


def clusterRTSP(dataset='target_points_pipes.csv', visualise=0, weights=None, visualise_speed=1.0, qhome=None, kmin=3, kmax=40, vel_limits=0.5, acc=0.5):
    """
    :param dataset: filename of input target points stored in /data/ package subdirectory
    :param visualise: 1 to display simulation otherwise 0
    :param weights: weights used for configuration selection
    :param visualise_speed: speed of visualisation (float)
    :param qhome: specify robot home position (array)
    :param kmin: min number of clusters if xmeans is selected (int)
    :param kmax: max number of clusters if xmeans is selected (int)
    :param vel_limits: desired maximum allowed velocity (as % of max.)
    :param acc: desired maximum acceleration (as % of max.)
    """

    starttime = timer()

    # ________________________________________________________
    ### DEFINE ENVIRONMENT
    # ________________________________________________________
    # initialise environment
    if qhome is None:
        qhome = [0, 0, 0, 0, 0, 0]

    env, robot, manipulator, iktype = environments.load_KUKA_kr6(display=0,  qhome=qhome, vel_limits=vel_limits, acc=acc)

    # ________________________________________________________
    ### INITIALISE
    # ________________________________________________________
    targets = []
    rospack = rospkg.RosPack()
    points = []
    path_to_files = rospack.get_path('cluster_rtsp')
    filename = str(path_to_files + '/data/'+dataset)
    with open(filename, 'rb') as csvfile:
        csv_points=csv.reader(csvfile, delimiter=',')
        for row in csv_points:
            points.append([float(i) for i in row])

    for i in xrange(0, len(points)):
        targets.append(orpy.Ray(np.array(points[i][0:3]), -1*np.array(points[i][3:])))

    n_points = len(targets)
    print('Number of points: %d' % n_points)

    initialise_end = timer()

    # set solver parameters
    params = classes.SolverParameters()
    params.standoff = 0.001
    params.step_size = np.pi / 3
    params.qhome = robot.GetActiveDOFValues()
    params.max_iters = 5000
    params.max_ppiters = 100
    params.try_swap = False
    configurations, ik_cpu_time = kinematics.compute_robot_configurations(env, robot, targets, params)

    # ________________________________________________________
    ### FORMAT CONFIG LIST
    # ________________________________________________________

    # append unique ID to each configuration
    # all_configs format: [j1 j2 j3 j4 j5 j6 cluster_n x y z task_point_ID config_ID]
    row = range(0, len(configurations))
    all_configs = np.column_stack((configurations[:, 0:6], row, configurations[:, 6:10], row))
    home_config = np.hstack((qhome, np.array([0, 0, 0, 0, all_configs[-1, 10] + 1, all_configs[-1, 11] + 1])))
    all_configs = np.vstack((all_configs, home_config))

    # ________________________________________________________
    ### CONFIGURATION SELECTIONS
    # ________________________________________________________

    print('Starting configuration selection...')
    if weights is None:
        weights = [0.2676, 0.3232, 0.2576, 0.0303, 0.0917, 0.0296]
    selected_configurations, select_time = cselect.clusterConfigSelection(all_configs, qhome, weights)

    # ________________________________________________________
    ### APPLY CLUSTERING METHOD TO CONFIG LIST
    # ________________________________________________________
    cluster_start = timer()
    print ('Clustering configurations...')
    xmeans = XMeans.fit(selected_configurations[:, 0:6], kmax=kmax, kmin=kmin, weights=np.array(weights)*6)
    N = xmeans.k
    labels = xmeans.labels_

    print('Number of clusters assigned: %d.' % N)

    # append cluster number to end of configuration points
    for i in xrange(0, len(selected_configurations)):
        selected_configurations[i, 6] = int(labels[i])

    # sort rows in ascending order based on 7th element
    ind = np.argsort(selected_configurations[:, 6])
    selected_configurations = selected_configurations[ind]

    cluster_end = timer()

    # ________________________________________________________
    ### GLOBAL TSP COMPUTATIONS
    # ________________________________________________________

    # Generate new variable for local clusters of points
    clusters = [None] * N
    for i in xrange(0, N):
        cluster = np.where(selected_configurations[:, 6] == i)[0]
        clusters[i] = selected_configurations[cluster, :]

    globsequence_start = timer()
    print('Computing global sequence...')
    gtour, pairs, closest_points, entry_points = tsp.globalTSP(clusters, qhome)
    global_path = gtour[1:-1]
    entry_points = entry_points[1:-1]
    globsequence_end = timer()

    # ________________________________________________________
    ### LOCAL TSP COMPUTATIONS
    # ________________________________________________________

    path = [None] * N
    print('Solving TSP for each cluster...')
    localtsp_start = timer()

    # plan intra-cluster paths
    for i in xrange(0, N):
        if np.shape(clusters[global_path[i]])[0] > 1:
            # Run Two-Opt
            tgraph = tsp.construct_tgraph(clusters[global_path[i]][:, 0:6], distfn=tsp.euclidean_fn)
            path[i] = tsp.two_opt(tgraph, start=entry_points[i][0], end=entry_points[i][1])
        else:
            path[i] = [0]

    localtsp_end = timer()

    # ________________________________________________________
    ### PLAN PATHS BETWEEN ALL POINTS IN COMPUTED PATH
    # ________________________________________________________

    # need to correct color indexing - somehow identify cluster index
    plan_start = timer()
    robot.SetActiveDOFValues(qhome)
    if N == 1:
        c = np.array([1, 0, 0])
    elif N < 10:
        c = np.array([[0.00457608, 0.58586408, 0.09916249],
                      [0.26603989, 0.36651324, 0.64662435],
                      [0.88546289, 0.63658585, 0.75394724],
                      [0.29854082, 0.26499636, 0.20025494],
                      [0.86513743, 0.98080264, 0.18520593],
                      [0.39864878, 0.33938585, 0.27366609],
                      [0.90286517, 0.51585244, 0.09724035],
                      [0.55158651, 0.56320824, 0.44465467],
                      [0.57776588, 0.38423542, 0.59291004],
                      [0.21227011, 0.9159966, 0.59002942]])
    else:
        c = np.random.rand(N, 3)

    h = []
    count = 0
    traj = [None] * (len(selected_configurations) + 1)
    skipped = 0

    points = [None] * len(selected_configurations)

    for i in xrange(0, N):
        idx = global_path[i]
        cluster = i + 1
        print ('Planning paths for configurations in cluster %i ...' % cluster)
        config = 0
        clock_start = timer()
        while config <= len(path[i]) - 1:
            q = clusters[idx][path[i][config], 0:6]

            traj[count] = ru.planning.plan_to_joint_configuration(robot, q, params.planner, params.max_iters, params.max_ppiters)
            if traj[count] is None:
                print ("Could not find a feasible path, skipping current point in cluster %i ... " % cluster)
                skipped += 1
                config += 1
                continue
            points[count] = np.hstack((clusters[idx][path[i][config], 7:10], clusters[idx][path[i][config], 6]))
            robot.SetActiveDOFValues(q)
            config += 1
            count += 1
            if config == len(path[i]):
                end_time = timer() - clock_start
                print('Planning time for cluster %d: %f' % (cluster, end_time))

    traj[-1] = ru.planning.plan_to_joint_configuration(robot, qhome, params.planner, params.max_iters, params.max_ppiters)
    robot.SetActiveDOFValues(qhome)

    # get time at end of planning execution
    end = timer()
    info = classes.InfoObj()

    info.initialise = initialise_end - starttime
    info.getconfig = ik_cpu_time
    info.configselection = select_time
    info.clustering = cluster_end - cluster_start
    info.globaltsp = globsequence_end - globsequence_start
    info.localtsp = localtsp_end - localtsp_start
    info.total_tsp = info.localtsp + info.globaltsp
    info.pathplanning = end - plan_start
    info.totalplanning = end - starttime
    info.execution = kinematics.compute_execution_time(traj)
    info.N_clusters = N
    info.n_points = len(selected_configurations) - skipped

    print('Initialisation time (s): %f' % info.initialise)
    print ('Get configurations time (s): %f' % info.getconfig)
    print ('Select configurations time (s): %f' % info.configselection)
    print ('Clustering time (s): %f' % info.clustering)
    print ('Compute global sequence time (s): %f' % info.globaltsp)
    print ('Local TSP time (s): %f' % info.localtsp)
    print ('Total TSP time (s): %f' % info.total_tsp)
    print ('Path planning time (s): %f' % info.pathplanning)
    print ('Total planning time (s): %f' % info.totalplanning)
    print ('Execution time (s): %f' % info.execution)
    print ('Number of visited points: %d' % info.n_points)
    # ________________________________________________________
    ### ACTUATE ROBOT
    # ________________________________________________________
    if visualise == 1:
        # Open display
        env.SetViewer('qtcoin')
        Tcamera = [[-0.75036157, 0.12281536, -0.6495182, 2.42751741],
                   [0.66099327, 0.12938928, -0.73915243, 2.34414649],
                   [-0.00673858, -0.98395874, -0.17826888, 1.44325936],
                   [0., 0., 0., 1.]]

        time.sleep(1)
        env.GetViewer().SetCamera(Tcamera)
        print('Starting simulation in 2 seconds...')
        time.sleep(2)

        start_exec = timer()
        cluster = 0
        count = 0
        for i in xrange(0, len(traj)):
            if traj[i] is None:
                continue
            if i <= len(points) - 1:
                cluster_next = points[i][3] + 1
            if cluster_next != cluster:
                cluster = cluster_next
                count += 1
                print ('Moving to configurations in cluster {:d} ...'.format(count))

            play_trajectory(env, robot, traj[i], visualise_speed)
            if i <= len(points) - 1:
                h.append(env.plot3(points=(points[i][0:3]), pointsize=4, colors=c[int(points[i][3])]))

        end_exec = timer() - start_exec
        print ('Simulation time: %f' % end_exec)

    raw_input('Press Enter to terminate...')

    return info


if __name__ == "__main__":
    HOME = [0, -np.pi / 3, (2 * np.pi) / 3, 0, -np.pi / 3, 0]
    clusterRTSP(visualise=1, qhome=HOME, vel_limits=0.75, acc=0.75, visualise_speed=4)
