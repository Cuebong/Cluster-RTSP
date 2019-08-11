# Cluster-RTSP
This ROS package provides an example implementation of the Cluster-RTSP algorithm for solving spatially-constrained robotic task sequencing problems. It has been tested on Ubuntu 14.04 and Ubuntu 16.04.

## Dependencies
This package was developed on OpenRave 0.9.0, which can be installed from this [GitHub page](https://github.com/crigroup/openrave-installation).

Installing Cluster-RTSP requires wstool. This can be installed using `sudo apt-get install python-wstool`.


## Installation
Create a new workspace, e.g.:

```
mkdir ~/catkin_ws
cd ~/catkin_ws
```

Initialize the workspace from the .rosinstall file and run catkin_make:
```
wstool init src https://raw.githubusercontent.com/Cuebong/Cluster-RTSP/master/clusterrtsp.rosinstall?token=AIGY25PHKSNQD6U65M2ZMTK5KBPPK
wstool update -t src
catkin_make
```

## Running the example
The example can be run using the _run\_clusterRTSP_ file in the ./scripts subdirectory.

Make the file executable:
```
chmod +x ./src/Cluster-RTSP/cluster_rtsp/scripts/run_clusterRTSP
```

Start roscore:
```
roscore
```

Run the example using rosrun (make sure you have sourced the setup.bash file using `source ./devel/setup.bash`). From a new terminal:

```
rosrun cluster_rtsp run_clusterRTSP
```
