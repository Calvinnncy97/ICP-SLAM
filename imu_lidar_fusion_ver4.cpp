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
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/ia_kfpcs.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>

#include <Eigen/Dense>

#include <iostream>
#include <fstream>

using namespace Eigen;

void lidarOdomCallback(sensor_msgs::PointCloud2::ConstPtr inputCloud);
void imuCallback(sensor_msgs::Imu::ConstPtr inputImu);

int corrSize = 0;

double initialTime = 0;
float initialXVelo = 0;
float initialYVelo = 0;
float initialZVelo = 0;

float totalYaw = 0;
float totalPitch = 0;
float totalRoll = 0;

float initialYaw = 0;
float initialPitch = 0;
float initialRoll = 0;

float prevX = 0;
float totalX = 0;
float count = 0;
int scalingCount = 0;

pcl::PointCloud<pcl::PointXYZ>::Ptr prevCloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::Normal>::Ptr prevCloudNormals (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::FPFHSignature33>::Ptr prevCloudFeatures (new pcl::PointCloud<pcl::FPFHSignature33>());

Matrix4f imuTransform = Matrix4f::Identity();
Matrix4f initialPose = Matrix4f::Identity();
Matrix4f prevTransform = Matrix4f::Identity();

std::list<Eigen::Matrix4f> imuBuffer;
std::list<float> xBuffer;

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

  ros::Subscriber imuSub = n.subscribe("/kitti/oxts/imu", 1500, imuCallback);
  ros::Subscriber pointCloudSub = n.subscribe("/kitti/velo/pointcloud", 1500, lidarOdomCallback);

  currentCloudPub = n.advertise<sensor_msgs::PointCloud2>("currentCloud", 10, true);
  prevCloudPub = n.advertise<sensor_msgs::PointCloud2>("prevCloud", 10, true);
  imuCloudPub = n.advertise<sensor_msgs::PointCloud2>("imuCloud", 10, true);
  alignedCloudPub = n.advertise<sensor_msgs::PointCloud2>("alignedCloud", 10, true);

  //ros::spin();

  ros::AsyncSpinner spinner(4);
  spinner.start();
  ros::waitForShutdown();

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
  voxel_grid.setLeafSize (0.5, 0.5, 0.5);
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

  pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr imuCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr kfpcsCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr sciaCloud (new pcl::PointCloud<pcl::PointXYZ>);

  Eigen::Matrix4f currentPose;
  Eigen::Matrix4f kfpcsTransform;
  Eigen::Matrix4f sciaTransform;

  // std::cout << "Points in source: " << prevCloud->size() << std::endl;
  // std::cout << "Points in target: " << filteredCloud->size() << std::endl;

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEst;
  normalEst.setInputCloud(filteredCloud);
  normalEst.useSensorOriginAsViewPoint();
  pcl::search::KdTree <pcl::PointXYZ>::Ptr searchTree (new pcl::search::KdTree <pcl::PointXYZ>());
  normalEst.setSearchMethod(searchTree);
  normalEst.setRadiusSearch(0.5);
  pcl::PointCloud<pcl::Normal>::Ptr filteredCloudNormals (new pcl::PointCloud<pcl::Normal>);
  normalEst.compute(*filteredCloudNormals);

  std::cout << "Finding Features" << std::endl;
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud(filteredCloud);
  fpfh.setInputNormals(filteredCloudNormals);
  pcl::search::KdTree <pcl::PointXYZ>::Ptr fpfhSearchTree (new pcl::search::KdTree <pcl::PointXYZ>());
  fpfh.setSearchMethod(fpfhSearchTree);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr filteredCloudFeatures (new pcl::PointCloud<pcl::FPFHSignature33>());
  fpfh.setRadiusSearch(0.5);
  fpfh.compute(*filteredCloudFeatures);

  if (prevCloud->size() > 0 && imuBuffer.size() > 0)
  {
    pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> est;
    est.setInputSource (filteredCloud);
    est.setInputTarget (prevCloud);

    pcl::CorrespondencesPtr cor (new pcl::Correspondences);
    est.determineCorrespondences (*cor, 0.5);

    float overlap = float(cor->size())/((filteredCloud->size()+prevCloud->size())/2);
    std::cout << "Overlap: " << overlap << std::endl;

    Matrix4f currentImuTransform = imuBuffer.front();
    imuBuffer.pop_front();

    Matrix4f icpTransform = Matrix4f::Identity();

    if (overlap < 0.60)
    {
      ros::Time startTime = ros::Time::now();

      std::cout << "FPCS Alignment" << std::endl;

      pcl::registration::FPCSInitialAlignment <pcl::PointXYZ, pcl::PointXYZ> kfpcs_ia; //this line can be changed to  pcl::registration::KFPCSInitialAlignment <pcl::PointXYZ, pcl::PointXYZ> to do keypoint-based 4-point congruent set alignment
    
      kfpcs_ia.setInputSource(filteredCloud);
      kfpcs_ia.setSourceNormals(filteredCloudNormals);

      kfpcs_ia.setInputTarget(prevCloud);
      kfpcs_ia.setTargetNormals(prevCloudNormals);

      kfpcs_ia.setMaxCorrespondenceDistance(0.5);
      kfpcs_ia.setMaxNormalDifference(0.2);
      // kfpcs_ia.setUpperTranslationThreshold(1.0);
      // kfpcs_ia.setLowerTranslationThreshold(0.1);

      kfpcs_ia.setApproxOverlap(overlap);
      kfpcs_ia.setDelta(0.01);
      kfpcs_ia.setMaximumIterations (50);
      kfpcs_ia.setNumberOfSamples(200);
      //kfpcs_ia.setTransformationEpsilon (0.0001);
      kfpcs_ia.setScoreThreshold(0.2);
      
      kfpcs_ia.align(*kfpcsCloud, currentImuTransform);
      kfpcsTransform = kfpcs_ia.getFinalTransformation();

      pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
      icp.setInputSource(filteredCloud);
      icp.setInputTarget(prevCloud);
    
      icp.setMaxCorrespondenceDistance (1.0);
      icp.setTransformationEpsilon (1e-6); //gives good score
      icp.setMaximumIterations (1e4);
      icp.setEuclideanFitnessEpsilon (0.01);

      icp.align(*alignedCloud, kfpcsTransform); 
		
      ros::Time endTime = ros::Time::now();
      ros::Duration fpcsInterval = endTime - startTime;

      std::cout << "FPCS Interval: " << fpcsInterval << std::endl;

      if (icp.hasConverged())
      {
        icpTransform = icp.getFinalTransformation();

        currentPose.block(0,0,3,3) = initialPose.block(0,0,3,3) * currentImuTransform.block(0,0,3,3);

        Vector3f sensorTranslation;
        sensorTranslation[0] = icpTransform(0,3);
        sensorTranslation[1] = icpTransform(1,3);
        sensorTranslation[2] = icpTransform(2,3);

        Vector3f worldTranslation = currentPose.block(0,0,3,3)*sensorTranslation;

        currentPose(0,3) = worldTranslation[0] + initialPose(0,3);
        currentPose(1,3) = worldTranslation[1] + initialPose(1,3);
        currentPose(2,3) = worldTranslation[2] + initialPose(2,3); 
      }

      endTime = ros::Time::now();
      ros::Duration fpcsIcpInterval = endTime - startTime - fpcsInterval;

      std::cout << "FPCS Interval: " << fpcsIcpInterval << std::endl;
    }
    else
    {
      ros::Time startTime = ros::Time::now();

      pcl::transformPointCloud (*prevCloud, *imuCloud, currentImuTransform);

      std::cout << "ICP Alignment" << std::endl;

      pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
      icp.setInputSource(filteredCloud);
      icp.setInputTarget(prevCloud);
    
      icp.setMaxCorrespondenceDistance (1.0);
      icp.setTransformationEpsilon (1e-9); //gives good score
      icp.setMaximumIterations (1e4);
      icp.setEuclideanFitnessEpsilon (0.001);

      //currentImuTransform(0,3) = currentImuTransform(0,3)/2;

      icp.align(*alignedCloud, currentImuTransform); 

      if (icp.hasConverged())
      {
        icpTransform = icp.getFinalTransformation();

        currentPose.block(0,0,3,3) = initialPose.block(0,0,3,3) * currentImuTransform.block(0,0,3,3);

        Vector3f sensorTranslation;
        sensorTranslation[0] = icpTransform(0,3);//+currentImuTransform(0,3)*2;
        sensorTranslation[1] = icpTransform(1,3);
        sensorTranslation[2] = icpTransform(2,3);

        Vector3f worldTranslation = currentPose.block(0,0,3,3)*sensorTranslation;

        currentPose(0,3) = worldTranslation[0] + initialPose(0,3);
        currentPose(1,3) = worldTranslation[1] + initialPose(1,3);
        currentPose(2,3) = worldTranslation[2] + initialPose(2,3); 
      }

      ros::Time endTime = ros::Time::now();
      ros::Duration icpInterval = endTime - startTime;

      std::cout << "ICP Interval: " << icpInterval << std::endl;
    }
    
    // pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
    // scia.setInputSource(filteredCloud);
    // scia.setSourceFeatures(filteredCloudFeatures);

    // scia.setInputTarget(prevCloud);
    // scia.setTargetFeatures(prevCloudFeatures);

    // std::cout << "Number of samples: " << scia.getNumberOfSamples() << std::endl; 

    // scia.align(*sciaCloud, currentImuTransform);
    // sciaTransform = scia.getFinalTransformation();

    // Matrix4f prototypeTransform  = Matrix4f::Identity();
    // prototypeTransform.block(0,0,3,3) = prevTransform.block(0,0,3,3) * currentImuTransform.block(0,0,3,3);
    // prototypeTransform.col(3) = prevTransform.col(3) + currentImuTransform.col(3);
    // prototypeTransform(3,3) = 1;

    //Matrix4f prototypeTransform  = Matrix4f::Identity();
    // prototypeTransform.block(0,0,3,3) = prevTransform.block(0,0,3,3) * currentImuTransform.block(0,0,3,3);
    // prototypeTransform.col(3) = prevTransform.col(3) + currentImuTransfor0.col(3currentImuTransform(0,3)/2;
    //prototypeTransform(0,3) = currentImuTransform(0,3);

      // std::cout << "Initial Pose: " << initialPose << std::endl;
      std::cout << "Current Pose: " << currentPose << std::endl;
      // std::cout << "Fitness Score: " << icp.getFitnessScore() << std::endl;
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
      
      initialPose = currentPose;
  }

    sensor_msgs::PointCloud2 filteredCloudMsg;
    sensor_msgs::PointCloud2 prevCloudMsg;
    sensor_msgs::PointCloud2 imuCloudMsg;
    sensor_msgs::PointCloud2 alignedCloudMsg;

    pcl::toROSMsg(*filteredCloud.get(), filteredCloudMsg);
    pcl::toROSMsg(*prevCloud.get(), prevCloudMsg);
    pcl::toROSMsg(*kfpcsCloud.get(), imuCloudMsg);
    pcl::toROSMsg(*alignedCloud.get(), alignedCloudMsg);

    currentCloudPub.publish(filteredCloudMsg);
    prevCloudPub.publish(prevCloudMsg);
    imuCloudPub.publish(imuCloudMsg);
    alignedCloudPub.publish(alignedCloudMsg);

    prevCloud->clear();
    pcl::copyPointCloud(*filteredCloud,*prevCloud);
    pcl::copyPointCloud(*filteredCloudNormals,*prevCloudNormals);
    pcl::copyPointCloud(*filteredCloudFeatures,*prevCloudFeatures);
}

void imuCallback (sensor_msgs::Imu::ConstPtr inputImu)
{
  float yawAngle, pitchAngle, rollAngle; //axis z,y,x
  float accX, accY, accZ;
  float x, y, z;

  double currentTime = inputImu -> header.stamp.sec + inputImu -> header.stamp.nsec / pow(10,9);
  if (initialTime != 0)
  {
    float interval = float(currentTime - initialTime); 
    // std::cout << "Interval = " << interval << std::endl;

    yawAngle = inputImu->angular_velocity.z * interval; //z angle
    pitchAngle = inputImu->angular_velocity.y * interval * -1; //y angle
    rollAngle = inputImu->angular_velocity.x * interval; //x angle

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

    accX = inputImu->linear_acceleration.x - sin(pitchAngle) * 9.81; 
    accY = inputImu->linear_acceleration.y + sin(rollAngle) * cos(pitchAngle) * 9.81; 
    accZ = inputImu->linear_acceleration.z + cos(rollAngle) * cos(pitchAngle) * 9.81; 

    x = (initialXVelo * interval + 0.5 * accX * pow(interval, 2));
    y = (initialYVelo * interval + 0.5 * accY * pow(interval, 2));
    z = (initialZVelo * interval + 0.5 * accZ * pow(interval, 2));


    // Vector3f sensorFrameTrans;
    // sensorFrameTrans[0] = x;
    // sensorFrameTrans[1] = y;
    // sensorFrameTrans[2] = z;

    // Matrix3f currentYaw = Matrix3f::Identity();
    // Matrix3f currentPitch = Matrix3f::Identity();
    // Matrix3f currentRoll = Matrix3f::Identity();

    // totalYaw = initialYaw + yawAngle;
    // totalPitch = initialPitch * pitchAngle;
    // totalRoll = initialRoll * rollAngle;

    // currentYaw(0,0) = cos(totalYaw);
    // currentYaw(0,1) = -sin(totalYaw);
    // currentYaw(1,0) = sin(totalYaw);
    // currentYaw(1,1) = cos(totalYaw);

    // currentPitch(0,0) = cos(totalPitch);
    // currentPitch(0,2) = sin(totalPitch);
    // currentPitch(2,0) = -sin(totalPitch);
    // currentPitch(2,2) = cos(totalPitch);

    // currentRoll(1,1) = cos(totalRoll);
    // currentRoll(1,2) = -sin(totalRoll);
    // currentRoll(2,1) = sin(totalRoll);
    // currentRoll(2,2) = cos(totalRoll);

    // Matrix3f currentRotationMatrix = currentYaw*currentPitch*currentRoll;

    // Vector3f navFrameTrans = currentRotationMatrix.inverse() * sensorFrameTrans;

    float weight = 0.5;
    float invWeight = 1 - weight;

    //x = prevX*weight + x*invWeight;

    // while (xBuffer.size() >= 50)
    // {
    //   xBuffer.pop_front();
    // }
    // xBuffer.push_back(x);

    //  totalX = std::accumulate(std::begin(xBuffer), std::end(xBuffer), 0.0);
    // count = xBuffer.size();
    // x = (totalX/count);

    //IMU to ICP input conversion
    Vector4f translation;
    translation[0] = x;
    translation[1] = 0; 
    translation[2] = 0;
    translation[3] = 1; 

    imuTransform.block(0,0,3,3) = rotationMatrix; 
    imuTransform.col(3) = translation;

    imuBuffer.push_back(std::move(imuTransform));

    initialXVelo += accX * interval;
    initialYVelo += accY * interval;
    initialZVelo += accZ * interval;

    initialYaw = totalYaw;
    initialPitch = totalPitch;
    initialRoll = totalRoll;

    prevX = x;
  }
  initialTime = currentTime;
}
