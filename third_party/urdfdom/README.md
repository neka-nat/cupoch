urdfdom
===========

The URDF (U-Robot Description Format) library provides core data structures and a simple XML parsers for populating the class data structures from an URDF file.

For now, the details of the URDF specifications reside on http://ros.org/wiki/urdf
  
### Build Status
[![Build Status](https://travis-ci.org/ros/urdfdom.png)](https://travis-ci.org/ros/urdfdom)

### Using with ROS

If you choose to check this repository out for use with ROS, be aware that the necessary ``package.xml`` is not 
included in this repo but instead is added in during the ROS release process. To emulate this, pull the appropriate
file into this repository using the following format. Be sure to replace the ALLCAPS words with the apropriate terms:

```
wget https://raw.github.com/ros-gbp/urdfdom-release/debian/ROS_DISTRO/UBUNTU_DISTRO/urdfdom/package.xml
```

For example:
```
wget https://raw.github.com/ros-gbp/urdfdom-release/debian/hydro/precise/urdfdom/package.xml
```

### Installing from Source with ROS Debians

**Warning: this will break ABI compatibility with future /opt/ros updates through the debian package manager. This is a hack, use at your own risk.**

If you want to install urdfdom from source, but not install all of ROS from source, you can follow these loose guidelines.
This is not best practice for installing, but works.
This version is for ROS Hydro but should be easily customized for future version of ROS:

```
sudo mv /opt/ros/hydro/include/urdf_parser/ /opt/ros/hydro/include/_urdf_parser/
sudo mv /opt/ros/hydro/lib/liburdfdom_model.so /opt/ros/hydro/lib/_liburdfdom_model.so
sudo mv /opt/ros/hydro/lib/liburdfdom_model_state.so /opt/ros/hydro/lib/_liburdfdom_model_state.so
sudo mv /opt/ros/hydro/lib/liburdfdom_sensor.so /opt/ros/hydro/lib/_liburdfdom_sensor.so
sudo mv /opt/ros/hydro/lib/liburdfdom_world.so /opt/ros/hydro/lib/_liburdfdom_world.so
sudo mv /opt/ros/hydro/lib/pkgconfig/urdfdom.pc /opt/ros/hydro/lib/pkgconfig/_urdfdom.pc
sudo mv /opt/ros/hydro/share/urdfdom/cmake/urdfdom-config.cmake /opt/ros/hydro/share/urdfdom/cmake/_urdfdom-config.cmake

cd ~/ros/urdfdom # where the git repo was checked out
mkdir -p build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=/opt/ros/hydro
make -j8
sudo make install

# now rebuild your catkin workspace
cd ~/ros/ws_catkin
catkin_make
```
