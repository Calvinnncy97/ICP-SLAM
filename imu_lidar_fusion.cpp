#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Imu.h"

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>

#include <Eigen/Dense>

#include <iostream>
#include <fstream>

using namespace Eigen;

void lidarOdomCallback(sensor_msgs::PointCloud2::ConstPtr inputCloud);
void imuCallback(sensor_msgs::Imu::ConstPtr inputImu);

double initialTime = 0;
float initialXVelo = 0;
float initialYVelo = 0;
float initialZVelo = 0;

bool imuFlag = true;

pcl::PointCloud<pcl::PointXYZ>::Ptr prevCloud (new pcl::PointCloud<pcl::PointXYZ>);

Matrix4f imuTransform = Matrix4f::Zero();
Matrix4f initialPose = Matrix4f::Identity();

std::list<Eigen::Matrix4f> imuBuffer;

int divergeCount = 0;

ros::Publisher currentCloudPub;
ros::Publisher prevCloudPub;
ros::Publisher imuCloudPub;
ros::Publisher alignedCloudPub;

int main(int argc, char **argv)
{
  std::cout << "Starting Subscriber" << std::endl;
  ros::init(argc, argv, "imu_lidar_fusion");
  ros::NodeHandle n;

  ros::Subscriber imuSub = n.subscribe("/kitti/oxts/imu", 10, imuCallback);
  ros::Subscriber pointCloudSub = n.subscribe("/kitti/velo/pointcloud", 10, lidarOdomCallback);

  currentCloudPub = n.advertise<sensor_msgs::PointCloud2>("currentCloud", 10, true);
  prevCloudPub = n.advertise<sensor_msgs::PointCloud2>("prevCloud", 10, true);
  imuCloudPub = n.advertise<sensor_msgs::PointCloud2>("imuCloud", 10, true);
  alignedCloudPub = n.advertise<sensor_msgs::PointCloud2>("alignedCloud", 10, true);
  ros::spin();

  return 0;
}

void lidarOdomCallback(sensor_msgs::PointCloud2::ConstPtr inputCloud)
{
  std::cout << "Starting ICP" << std::endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*inputCloud, *filteredCloud);

  double maxRange = std::numeric_limits<double>::max();
  double minRange = -std::numeric_limits<double>::max();
  std::vector<int> index;
  pcl::removeNaNFromPointCloud(*filteredCloud, *filteredCloud, index);

  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setMinimumPointsNumberPerVoxel(10);
  voxel_grid.setInputCloud (filteredCloud);
  voxel_grid.setDownsampleAllData(false);
  voxel_grid.setLeafSize (0.8f, 0.8f, 0.8f);
  voxel_grid.filter(*filteredCloud);

  pcl::PassThrough<pcl::PointXYZ> pass_x;
  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(minRange, maxRange);
  pcl::PassThrough<pcl::PointXYZ> pass_y;
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(minRange, maxRange);
  pcl::PassThrough<pcl::PointXYZ> pass_z;
  pass_z.setFilterFieldName("z");
  pass_z.setFilterLimits(minRange, maxRange);

  pass_x.setInputCloud(filteredCloud->makeShared());
  pass_x.filter(*filteredCloud);
  pass_y.setInputCloud(filteredCloud->makeShared());
  pass_y.filter(*filteredCloud);
  pass_z.setInputCloud(filteredCloud->makeShared());
  pass_z.filter(*filteredCloud);
/*
  signed int pointDiff = filteredCloud->size() - prevCloud->size();
  pointDiff = abs(pointDiff);
  if (filteredCloud->size() > prevCloud->size())
  {
    for (int i=0; i<pointDiff; i++)
    {
      prevCloud->push_back(pcl::PointXYZ(0,0,0));
    }
  }
  else 
  {
    for (int i=0; i<pointDiff; i++)
    {
      filteredCloud->push_back(pcl::PointXYZ(0,0,0));
    }
  }
*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr imuCloud (new pcl::PointCloud<pcl::PointXYZ>);

  Eigen::Matrix4f currentPose;
  std::cout << "Points in source: " << prevCloud->size() << std::endl;
  std::cout << "Points in target: " << filteredCloud->size() << std::endl;
  if (prevCloud->size() > 0 && imuBuffer.size() > 0)
  {
    
    pcl::transformPointCloud (*prevCloud, *imuCloud, imuBuffer.front()); 
    std::cout << "IMU Transformation: " << imuBuffer.front() << std::endl;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(filteredCloud);
    icp.setInputTarget(prevCloud);
    
    icp.setMaxCorrespondenceDistance (0.8);
    //icp.setMaximumIterations (1e4); //gives good score
    icp.setTransformationEpsilon (1e-9); //gives good score
    icp.setMaximumIterations (1e5);
    icp.setEuclideanFitnessEpsilon (0.01);

    icp.align(*alignedCloud, imuBuffer.front());
    imuBuffer.pop_front();
    Eigen::Matrix4f icpTransform;
    if (icp.hasConverged())
    {
      icpTransform = icp.getFinalTransformation();
      
      float icpX = icpTransform(0,3);
      float icpY = icpTransform(1,3);
      float icpZ = icpTransform(2,3);
      //icpTransform(0,3) = icpZ;
      //icpTransform(2,3) = icpX;

      float icpYawAngle = atan(icpTransform(1,0)/icpTransform(0,0));
      float icpPitchAngle = atan((icpTransform(2,0)*-1)/pow(pow(icpTransform(2,1),2)+pow(icpTransform(2,2),2),0.5));
      float icpRollAngle = atan(icpTransform(2,1)/icpTransform(2,2));

      float yawAngle = icpPitchAngle;
      float pitchAngle = icpYawAngle * -1; //Kitti yaw
      float rollAngle =  icpRollAngle;

      Matrix3f yaw = Matrix3f::Identity(); 
      Matrix3f pitch = Matrix3f::Identity(); 
      Matrix3f roll = Matrix3f::Identity(); 

      yaw(0,0) = cos(yawAngle);
      yaw(0,1) = -sin(yawAngle);
      yaw(1,0) = sin(yawAngle);
      yaw(1,1) = cos(yawAngle);

      pitch(0,0) = cos(pitchAngle);
      pitch(0,2) = sin(pitchAngle);
      pitch(2,0) = -sin(pitchAngle);
      pitch(2,2) = cos(pitchAngle);

      roll(1,1) = cos(rollAngle);
      roll(1,2) = -sin(rollAngle);
      roll(2,1) = sin(rollAngle);
      roll(2,2) = cos(rollAngle);

      Matrix3f rotationMatrix = yaw*pitch*roll;

      Vector4f translation; 
      translation[0] = icpZ;
      translation[1] = icpY * -1;
      translation[2] = icpX;
      translation[3] = 1; 

      Matrix4f newIcpMatrix = Matrix4f::Zero();

      newIcpMatrix.block(0,0,3,3) = rotationMatrix; 
      newIcpMatrix.col(3) = translation;

      std::cout << "ICP Transformation: " << icpTransform << std::endl;
      Eigen::Matrix4f finalTransform = newIcpMatrix;

      std::cout << "Final Transformation: " << finalTransform << std::endl;
      currentPose = finalTransform * initialPose;
      std::cout << "Initial Pose: " << initialPose << std::endl;
      std::cout << "Current Pose: " << currentPose << std::endl;
      std::cout << "Fitness Score: " << icp.getFitnessScore() << std::endl;
      std::ofstream file;
      file.open ("2011_10_03_drive_0027_sync.txt", std::ios::app);
      for (int i = 0; i <= currentPose.rows()-2; i++)
      {
       for (int j = 0; j <= currentPose.cols()-1; j++)
       {
           file << currentPose(i,j) << " ";
       }
      }
      file << std::endl;
    }
    else
    {
      divergeCount++;
      std::cout << "ICP did not converge" << std::endl;
      std::cout << "Divergent Count: " << divergeCount << std::endl;
    }

    for (int i=0; i<=3; i++)
    {
      for (int j=0; j<=3; j++)
      {
        initialPose(i,j) = currentPose(i,j);
      }
    }
    sensor_msgs::PointCloud2 filteredCloudMsg;
    sensor_msgs::PointCloud2 prevCloudMsg;
    sensor_msgs::PointCloud2 imuCloudMsg;
    sensor_msgs::PointCloud2 alignedCloudMsg;

    pcl::toROSMsg(*filteredCloud.get(), filteredCloudMsg);
    pcl::toROSMsg(*prevCloud.get(), prevCloudMsg);
    pcl::toROSMsg(*imuCloud.get(), imuCloudMsg);
    pcl::toROSMsg(*alignedCloud.get(), alignedCloudMsg);

    currentCloudPub.publish(filteredCloudMsg);
    prevCloudPub.publish(prevCloudMsg);
    imuCloudPub.publish(imuCloudMsg);
    alignedCloudPub.publish(alignedCloudMsg);
  }
  prevCloud->clear();
  pcl::copyPointCloud(*filteredCloud,*prevCloud);
  std::cout << "replicate pose" << std::endl;
}

void imuCallback (sensor_msgs::Imu::ConstPtr inputImu)
{
  imuFlag = true;
  float yawAngle, pitchAngle, rollAngle; //axis z,y,x
  float x, y, z;

  double currentTime = inputImu -> header.stamp.sec + inputImu -> header.stamp.nsec / pow(10,9);
  std::cout << "Current Time: " << currentTime << std::endl;
  if (initialTime != 0)
  {
    float interval = float(currentTime - initialTime); 
    std::cout << "Interval = " << interval << std::endl;

    yawAngle = inputImu->angular_velocity.y * interval; //z angle
    pitchAngle = inputImu->angular_velocity.z * interval * -1; //y angle
    rollAngle = inputImu->angular_velocity.x * interval * -1; //x angle

    Matrix3f yaw = Matrix3f::Identity(); 
    Matrix3f pitch = Matrix3f::Identity(); 
    Matrix3f roll = Matrix3f::Identity(); 

    yaw(0,0) = cos(yawAngle);
    yaw(0,1) = -sin(yawAngle);
    yaw(1,0) = sin(yawAngle);
    yaw(1,1) = cos(yawAngle);

    pitch(0,0) = cos(pitchAngle);
    pitch(0,2) = sin(pitchAngle);
    pitch(2,0) = -sin(pitchAngle);
    pitch(2,2) = cos(pitchAngle);

    roll(1,1) = cos(rollAngle);
    roll(1,2) = -sin(rollAngle);
    roll(2,1) = sin(rollAngle);
    roll(2,2) = cos(rollAngle);

    Matrix3f rotationMatrix = yaw*pitch*roll;

    y = (initialXVelo * interval + 0.5 * inputImu->linear_acceleration.x * pow(interval, 2)) * -1;
    x = (initialYVelo * interval + 0.5 * inputImu->linear_acceleration.y * pow(interval, 2)) * -1;
    z = (initialZVelo * interval + 0.5 * inputImu->linear_acceleration.z * pow(interval, 2)) * -1;

    // Vector4f translation;
    // translation[0] = x;
    // translation[1] = y;
    // translation[2] = z;
    // translation[3] = 1; 

    //IMU to ICP input conversion
    Vector4f translation;
    translation[0] = z;
    translation[1] = -y;
    translation[2] = x;
    translation[3] = 1; 

    imuTransform.block(0,0,3,3) = rotationMatrix; 
    imuTransform.col(3) = translation;

    imuBuffer.push_back(std::move(imuTransform));
    /*
    imuTransform(0,3) = x;
    imuTransform(1,3) = y;
    imuTransform(2,3) = z;
    imuTransform(3,3) = 1;
    */

    initialXVelo = x;
    initialYVelo = y;
    initialZVelo = z;
  }
  initialTime = currentTime;
  imuFlag = false;
}
