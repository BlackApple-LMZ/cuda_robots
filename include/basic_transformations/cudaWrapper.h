#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "basic_transformations.h"
#include <pcl/point_cloud.h>


class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();
    void printDeviceProp(const cudaDeviceProp &prop);

	bool rotateLeft(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
	bool rotateRight(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
	bool translateForward(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
	bool translateBackward(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
	bool removePointsInsideSphere(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
	bool transform(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3f matrix);

};



#endif
