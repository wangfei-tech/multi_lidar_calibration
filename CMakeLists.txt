cmake_minimum_required(VERSION 3.0.2)
project(multi_lidar_calibration)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2
  image_transport
)

find_package(PCL REQUIRED QUIET)


catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES multi_lidar_calibration
 CATKIN_DEPENDS pcl_ros sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Declare a C++ library
 add_library(multi_lidar_calibration
   src/multi_lidar_calibration.cc
 )


## Specify libraries to link a library or executable target against
 target_link_libraries(multi_lidar_calibration
   ${catkin_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES}
 )

add_executable(multi_lidar_calibration_node src/multi_lidar_calibration_node.cc)
target_link_libraries(multi_lidar_calibration_node
${catkin_LIBRARIES} multi_lidar_calibration
)
add_dependencies(multi_lidar_calibration_node multi_lidar_calibration )


## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(TARGETS multi_lidar_calibration multi_lidar_calibration_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.launch"
)

