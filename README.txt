Install dependencies
-PCL 1.11
-pcl-conversions (latest version)

Build these 2 packages from source and compile using the CMAKEList and package.xml. CMakeList contains some source code which you don't have, those can be removed.

The code can be run with a single rosrun. On the other hand, play the rosbag file of the Kitti data. You need to download the raw data from Kitti website and convert them to rosbag. 
