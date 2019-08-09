import openravepy as orpy
import time
import numpy as np
import rospkg


def load_ikfast(robot, iktype, autogenerate=True):
    ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(robot, iktype=iktype)
    if not ikmodel.load() and autogenerate:
        print 'Generating IKFast {0}. It will take few minutes...'.format(iktype.name)
        ikmodel.autogenerate()
        print 'IKFast {0} has been successfully generated'.format(iktype.name)
    return ikmodel.load()


def load_KUKA_kr6(display=1, qhome=None, vel_limits=0.5, acc=0.5):
    # Load the OpenRAVE environment
    env = orpy.Environment()

    rospack = rospkg.RosPack()
    env_name = rospack.get_path('kuka_openrave') + '/worlds/kuka_ERICA_hollow_pipe.env.xml'

    if not env.Load(env_name):
        print('Failed to load the world. Did you run: catkin_make install?')
        exit(1)

    if qhome is None:
        qhome = [0, 0, 0, 0, 0, 0]

    # Setup robot and manipulator
    robot = env.GetRobot('robot')
    manipulator = robot.SetActiveManipulator('generic_tool')

    robot.SetActiveDOFs(manipulator.GetArmIndices())

    # Set velocity and acceleration limits
    robot.SetDOFVelocityLimits(robot.GetDOFVelocityLimits() * vel_limits)
    robot.SetDOFAccelerationLimits(np.array([2.955, 3.03, 5.607, 6, 18.75, 3]) * acc)

    # Update joint limits
    joint_limits = np.radians(np.array([[-165, 165],
                                        [-185, 40],
                                        [-115, 151],
                                        [-180, 180],
                                        [-115, 115],
                                        [-345, 345]]))
    for i in range(0, 6):
        robot.GetJoints()[i].SetLimits(np.array([joint_limits[i, 0]]), np.array([joint_limits[i, 1]]))

    # Load IKfast and link statistics databases for finding close IK solutions
    iktype = orpy.IkParameterization.Type.Transform6D
    success = load_ikfast(robot, iktype)
    if not success:
        print('Failed to load IKFast for {0}, manipulator: {1}'.format(robot.GetName(), manipulator.GetName()))
        exit(1)
    statsmodel = orpy.databases.linkstatistics.LinkStatisticsModel(robot)
    if not statsmodel.load():
        print('Generating LinkStatistics database. It will take around 1 minute...')
        statsmodel.autogenerate()
    statsmodel.setRobotWeights()
    statsmodel.setRobotResolutions(xyzdelta=0.01)

    robot.SetActiveDOFValues(qhome)

    if display == 1:
        env.SetViewer('qtcoin')
        Tcamera = [[-0.75036157, 0.12281536, -0.6495182, 2.42751741],
                   [0.66099327, 0.12938928, -0.73915243, 2.34414649],
                   [-0.00673858, -0.98395874, -0.17826888, 1.44325936],
                   [0., 0., 0., 1.]]

        time.sleep(2)
        env.GetViewer().SetCamera(Tcamera)

        raw_input('Press Enter to close...')

    return env, robot, manipulator, iktype


if __name__ == "__main__":
    HOME = [0, -np.pi / 3, (2 * np.pi) / 3, 0, -np.pi / 3, 0]
    env, robot, manipulator, iktype = load_KUKA_kr6(qhome=HOME)
