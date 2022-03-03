cmake_minimum_required(VERSION 3.0.2)
project(lidar_slam)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set(PACKAGE_DEPENDENCIES
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  octomap_ros
  octomap_msgs
  eigen_conversions
  cmake_modules
) 

find_package(cmake_modules REQUIRED)

find_package(pcl_conversions)

find_package(catkin REQUIRED COMPONENTS ${PACKAGE_DEPENDENCIES})

find_package(octomap REQUIRED)

find_package(PCL 1.8 REQUIRED COMPONENTS common io search registration features filters)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_libraries(${PCL_LIBRARIES})


find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIBRARY_DIRS})
link_libraries(${G2O_LIBRARIES})

catkin_package(
  #INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS ${PACKAGE_DEPENDENCIES}
  DEPENDS OCTOMAP
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

include_directories(/usr/include/suitesparse/)

set(LINK_LIBS
  ${OCTOMAP_LIBRARIES}
  ${catkin_LIBRARIES}
  g2o_stuff 
  g2o_core
  cxsparse
  cholmod
)

#add_executable(imu_lidar_fusion src/imu_lidar_fusion.cpp)
#target_link_libraries(imu_lidar_fusion ${catkin_LIBRARIES})

#add_executable(imu_lidar_fusion_ver2 src/imu_lidar_fusion_ver2.cpp)
#target_link_libraries(imu_lidar_fusion_ver2 ${catkin_LIBRARIES})

#add_executable(imu_lidar_fusion_ver3 src/imu_lidar_fusion_ver3.cpp)
#target_link_libraries(imu_lidar_fusion_ver3 ${catkin_LIBRARIES})

add_executable(imu_lidar_fusion_ver4 src/imu_lidar_fusion_ver4.cpp)
target_link_libraries(imu_lidar_fusion_ver4 ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(imu_lidar_fusion_ver5 src/imu_lidar_fusion_ver5.cpp)
#target_link_libraries(imu_lidar_fusion_ver5 ${catkin_LIBRARIES})

#add_executable(imu_lidar_fusion_pcl_visualizer src/imu_lidar_fusion_pcl_visualizer.cpp)
#target_link_libraries(imu_lidar_fusion_pcl_visualizer ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(imu_lidar_fusion_ver6 src/imu_lidar_fusion_ver6.cpp)
#target_link_libraries(imu_lidar_fusion_ver6 ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(imu_lidar_fusion_ekf src/imu_lidar_fusion_ekf.cpp)
#target_link_libraries(imu_lidar_fusion_ekf ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(imu_lidar_fusion_default src/imu_lidar_fusion_default.cpp)
#target_link_libraries(imu_lidar_fusion_default ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(imu_lidar_fusion_FPCS src/imu_lidar_fusion_FPCS.cpp)
#target_link_libraries(imu_lidar_fusion_FPCS ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(imu_lidar_fusion_5hz src/imu_lidar_fusion_5hz.cpp)
#target_link_libraries(imu_lidar_fusion_5hz ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(optimization src/optimization.cpp)
#target_link_libraries(optimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${G2O_LIBRARIES} ${LINK_LIBS})

add_executable(own_icp_trunk src/own_icp_trunk.cpp)
target_link_libraries(own_icp_trunk ${catkin_LIBRARIES} ${PCL_LIBRARIES})


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/lidar_slam.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/lidar_slam_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_lidar_slam.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
