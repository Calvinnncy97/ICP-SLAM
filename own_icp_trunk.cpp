#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Imu.h"

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_distance.h>
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
pcl::PointCloud<pcl::PointNormal>::Ptr prevCloudNormals (new pcl::PointCloud<pcl::PointNormal>);
pcl::PointCloud<pcl::FPFHSignature33>::Ptr prevCloudFeatures (new pcl::PointCloud<pcl::FPFHSignature33>());

Matrix4f imuTransform = Matrix4f::Identity();
Matrix4f initialPose = Matrix4f::Identity();
Matrix4f prevTransform = Matrix4f::Identity();
Matrix4f edgeTransform = Matrix4f::Identity();

std::list<Eigen::Matrix4f> imuBuffer;
std::list<float> xBuffer;

int divergeCount = 0;

ros::Publisher currentCloudPub;
ros::Publisher prevCloudPub;
ros::Publisher imuCloudPub;
ros::Publisher alignedCloudPub;

bool waitForImu = true;

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <pcl::PointNormal>
{
  using pcl::PointRepresentation<pcl::PointNormal>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const pcl::PointNormal &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

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

  while (ros::ok())
  {
    ros::spinOnce();
  }

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
  voxel_grid.setMinimumPointsNumberPerVoxel(20);
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

  pcl::PointCloud<pcl::PointNormal>::Ptr alignedCloud (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr imuCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr kfpcsCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr sciaCloud (new pcl::PointCloud<pcl::PointXYZ>);

  Eigen::Matrix4f currentPose;
  Eigen::Matrix4f kfpcsTransform;
  Eigen::Matrix4f sciaTransform;

  std::cout << "Points in source: " << filteredCloud->size() << std::endl;
  std::cout << "Points in target: " << prevCloud->size() << std::endl;

  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> normalEst;
  normalEst.setInputCloud(filteredCloud);
  //normalEst.useSensorOriginAsViewPoint();
  normalEst.setViewPoint(0,0,0);
  pcl::search::KdTree <pcl::PointXYZ>::Ptr searchTree (new pcl::search::KdTree <pcl::PointXYZ>());
  normalEst.setSearchMethod(searchTree);
  normalEst.setKSearch(10);
  pcl::PointCloud<pcl::PointNormal>::Ptr filteredCloudNormals (new pcl::PointCloud<pcl::PointNormal>);
  normalEst.compute(*filteredCloudNormals);

  pcl::PointCloud<pcl::PointNormal>::Ptr sourceCloud (new pcl::PointCloud<pcl::PointNormal>);
  pcl::copyPointCloud(*filteredCloud, *sourceCloud);
  pcl::copyPointCloud(*filteredCloudNormals, *sourceCloud);
  pcl::copyPointCloud(*filteredCloud, *filteredCloudNormals);

  Vector3f worldTranslation;
  Vector3f sensorTranslation;

  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);


  if (prevCloud->size() > 0 && imuBuffer.size() > 0)
  {
    std::cout << "First buffer element : " << imuBuffer.front() << std::endl;
    std::cout << "Element transferred" << std::endl;
    bool pause = true;
    while (pause)
    {
      if (!waitForImu)
      {
        break;
      } 
      std::cout << "Waitinf for IMU" << std::endl;
    }
    Eigen::Matrix4f currentImuTransform = imuBuffer.front();
    imuBuffer.pop_front();
    std::cout << "Buffer size: " << imuBuffer.size() << std::endl;    

    Matrix4f icpTransform = Matrix4f::Identity();

    pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal> est;
    //std::cout << sourceCloud->size() << std::endl;
    pcl::PointCloud<pcl::PointNormal>::ConstPtr filteredCloudNormalsConstPtr (new pcl::PointCloud<pcl::PointNormal> (*filteredCloudNormals->makeShared()));
    pcl::PointCloud<pcl::PointNormal>::ConstPtr prevCloudNormalsConstPtr (new pcl::PointCloud<pcl::PointNormal> (*prevCloudNormals->makeShared()));
    est.setInputSource (filteredCloudNormalsConstPtr->makeShared());
    est.setInputTarget (prevCloudNormalsConstPtr->makeShared());

    pcl::CorrespondencesPtr cor (new pcl::Correspondences);
    est.determineCorrespondences (*cor, 2.0);

    pcl::registration::CorrespondenceRejectorDistance distRejector;
    distRejector.setMaximumDistance(1.0);
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr filteredCloudConstPtr (new pcl::PointCloud<pcl::PointXYZ> (*filteredCloud->makeShared()));
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr prevCloudConstPtr (new pcl::PointCloud<pcl::PointXYZ> (*prevCloud->makeShared()));
    distRejector.setInputSource<pcl::PointXYZ> (filteredCloudConstPtr->makeShared());
    distRejector.setInputTarget<pcl::PointXYZ> (prevCloudConstPtr->makeShared());
    distRejector.setInputCorrespondences (cor);
    //pcl::CorrespondencesPtr dist_cor (new pcl::Correspondences);
    //distRejector.getRemainingCorrespondences (*cor, *dist_cor);

    pcl::registration::CorrespondenceRejectorSurfaceNormal normRejector;
    normRejector.initializeDataContainer<pcl::PointNormal, pcl::PointNormal>();
    normRejector.setThreshold(0.5);
    normRejector.setInputSource<pcl::PointNormal>(filteredCloudNormalsConstPtr->makeShared());
    normRejector.setInputTarget<pcl::PointNormal>(filteredCloudNormalsConstPtr->makeShared());
    // pcl::CorrespondencesPtr final_cor (new pcl::Correspondences);
    // normRejector.getRemainingCorrespondences (*dist_cor, *final_cor);

    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
    icp.setEnforceSameDirectionNormals(true);
    icp.setInputSource(filteredCloudNormals->makeShared());
    icp.setInputTarget(prevCloudNormals->makeShared());
    icp.setMaxCorrespondenceDistance(1.0);

    MyPointRepresentation point_representation;
    //icp.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));
    pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal>::Ptr corrEstPtr (new pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal> (est));
    icp.setCorrespondenceEstimation(corrEstPtr);
    pcl::registration::CorrespondenceRejectorDistance::Ptr distRejectorPtr (new pcl::registration::CorrespondenceRejectorDistance (distRejector));
    pcl::registration::CorrespondenceRejectorSurfaceNormal::Ptr normRejectorPtr (new pcl::registration::CorrespondenceRejectorSurfaceNormal (normRejector));
    std::cout << "Normal Threshold: " << normRejectorPtr -> getThreshold() << std::endl;
    icp.addCorrespondenceRejector (distRejectorPtr);
    icp.addCorrespondenceRejector (normRejectorPtr);
    
    icp.setTransformationEpsilon (0.01); //gives good score
    icp.setMaximumIterations (100);
    icp.setTransformationRotationEpsilon (0.9999995);
    //icp.setEuclideanFitnessEpsilon(0.01);
    //alignedCloud = filteredCloudNormals;
    icp.align(*alignedCloud, currentImuTransform); 

    if (icp.hasConverged())
    {
      icpTransform = icp.getFinalTransformation();

      currentPose.block(0,0,3,3) = initialPose.block(0,0,3,3) * currentImuTransform.block(0,0,3,3);

      sensorTranslation[0] = icpTransform(0,3);//+currentImuTransform(0,3)*2;
      sensorTranslation[1] = icpTransform(1,3);
      sensorTranslation[2] = icpTransform(2,3);

      worldTranslation = initialPose.block(0,0,3,3)*sensorTranslation;

      currentPose(0,3) = worldTranslation[0] + initialPose(0,3);
      currentPose(1,3) = worldTranslation[1] + initialPose(1,3);
      currentPose(2,3) = worldTranslation[2] + initialPose(2,3); 
    } 

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
    prevCloudNormals->clear();
    pcl::copyPointCloud(*filteredCloud,*prevCloud);
    pcl::copyPointCloud(*filteredCloudNormals,*prevCloudNormals);
    // pcl::copyPointCloud(*filteredCloudFeatures,*prevCloudFeatures);
    waitForImu = true;
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

    Vector4f translation;
    translation[0] = 0.75;
    translation[1] = 0; 
    translation[2] = 0;
    translation[3] = 1; 

    imuTransform.block(0,0,3,3) = rotationMatrix; 
    imuTransform.col(3) = translation;

    imuBuffer.push_back(imuTransform);
    waitForImu = false;
    

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
