#include "basic_transformations/cudaWrapper.h"


CCudaWrapper::CCudaWrapper()
{

}

CCudaWrapper::~CCudaWrapper()
{

}

void CCudaWrapper::warmUpGPU()
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return;
	err = cudaWarmUpGPU();
		if(err != ::cudaSuccess)return;

}
void CCudaWrapper::printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem); //全局内存总量
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock); //每个线程块可用的共享内存总量
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);  //每一个线程块上可用的32位寄存器数量
    printf("warpSize : %d.\n", prop.warpSize); //一个线程束包含的线程数量，在实际运行中，线程块会被分割成更小的线程束(warp)，线程束中的每个线程都将在不同数据上执行相同的命令
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock); //每一个线程块中支持的最大线程数量
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //每一个线程块的每个维度的最大大小(x,y,z)
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); //每一个线程格的每个维度的最大大小(x,y,z)
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor); //设备计算能力主版本号 次版本号
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}
int CCudaWrapper::getNumberOfAvailableThreads()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0); //获取设备属性

	int threads = 0;
    //printDeviceProp(prop);
	if(prop.major == 2)
	{
		threads=prop.maxThreadsPerBlock/2;
	}else if(prop.major > 2)
	{
		threads=prop.maxThreadsPerBlock;
	}else
	{
		return 0;
	}
    //std::cout<<threads<<std::endl;
	return threads;
}

bool CCudaWrapper::rotateLeft(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	float anglaRad = 10.0f*M_PI/180.0;

	Eigen::Affine3f mr;
			mr = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(anglaRad, Eigen::Vector3f::UnitZ());

	if(!transform(point_cloud, mr))
	{
		std::cout << "Problem with transform" << std::endl;
		cudaDeviceReset();
		return false;
	}
	return true;
}

bool CCudaWrapper::rotateRight(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	float anglaRad = -10.0f*M_PI/180.0;

	Eigen::Affine3f mr;
			mr = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(anglaRad, Eigen::Vector3f::UnitZ());

	if(!transform(point_cloud, mr))
	{
		std::cout << "Problem with transform" << std::endl;
		cudaDeviceReset();
		return false;
	}
	return true;
}

bool CCudaWrapper::translateForward(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	//Eigen::Affine3f mt = Eigen::Affine3f::Identity();
	//mt(0,3) = 1.0f;
	Eigen::Affine3f mt(Eigen::Translation3f(Eigen::Vector3f(1.0f, 0.0f, 0.0f)));
	if(!transform(point_cloud, mt))
	{
		std::cout << "Problem with transform" << std::endl;
		cudaDeviceReset();
		return false;
	}
	return true;
}

bool CCudaWrapper::translateBackward(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	Eigen::Affine3f mt(Eigen::Translation3f(Eigen::Vector3f(-1.0f, 0.0f, 0.0f)));
	if(!transform(point_cloud, mt))
	{
		std::cout << "Problem with transform" << std::endl;
		cudaDeviceReset();
		return false;
	}
	return true;
}

bool CCudaWrapper::removePointsInsideSphere(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	int threads;
	float sphere_radius = 1.0f;
	pcl::PointXYZ * d_point_cloud;
	bool* d_markers;
	bool* h_markers;

	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	threads = getNumberOfAvailableThreads();

	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointXYZ) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_markers, point_cloud.points.size()*sizeof(bool) );
				if(err != ::cudaSuccess)return false;


	err = cudaRemovePointsInsideSphere(threads, d_point_cloud, d_markers, point_cloud.points.size(), sphere_radius);
		if(err != ::cudaSuccess)return false;

	h_markers = (bool *)malloc(point_cloud.points.size()*sizeof(bool));

	err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size()*sizeof(bool),cudaMemcpyDeviceToHost);
				if(err != ::cudaSuccess)return false;

	pcl::PointCloud<pcl::PointXYZ> new_point_cloud;
	for(size_t i = 0; i < point_cloud.points.size(); i++)
	{
		if(h_markers[i])new_point_cloud.push_back(point_cloud[i]);
	}

	std::cout << "Number of points before removing points: " << point_cloud.size() << std::endl;
	point_cloud = new_point_cloud;
	std::cout << "Number of points after removing points: " << point_cloud.size() << std::endl;



	free(h_markers);

	err = cudaFree(d_markers); d_markers = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	return true;
}
//point_cloud 这个指针指向host的点云空间 d_point_cloud这个指针指向device的点云空间
bool CCudaWrapper::transform(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3f matrix)
{
	int threads;
	pcl::PointXYZ * d_point_cloud;

	float h_m[16]; //给host分配变换矩阵的内存
	float *d_m;

	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	threads = getNumberOfAvailableThreads();

	h_m[0] = matrix.matrix()(0,0);
	h_m[1] = matrix.matrix()(1,0);
	h_m[2] = matrix.matrix()(2,0);
	h_m[3] = matrix.matrix()(3,0);

	h_m[4] = matrix.matrix()(0,1);
	h_m[5] = matrix.matrix()(1,1);
	h_m[6] = matrix.matrix()(2,1);
	h_m[7] = matrix.matrix()(3,1);

	h_m[8] = matrix.matrix()(0,2);
	h_m[9] = matrix.matrix()(1,2);
	h_m[10] = matrix.matrix()(2,2);
	h_m[11] = matrix.matrix()(3,2);

	h_m[12] = matrix.matrix()(0,3);
	h_m[13] = matrix.matrix()(1,3);
	h_m[14] = matrix.matrix()(2,3);
	h_m[15] = matrix.matrix()(3,3);

	err = cudaMalloc((void**)&d_m, 16*sizeof(float) ); //给device分配变换矩阵的内存
		if(err != ::cudaSuccess)return false;


	err = cudaMemcpy(d_m, h_m, 16*sizeof(float), cudaMemcpyHostToDevice); //给device分配将变换矩阵数据由host拷贝到device
		if(err != ::cudaSuccess)return false;

	//std::cout<<h_m[0]<<" "<<h_m[8]<<" "<<h_m[15]<<std::endl; //复制都是浅拷贝 不会清除原来的数据

	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointXYZ) );//给device分配点云的内存
			if(err != ::cudaSuccess)return false;

    //把点云数据由host复制到device
	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	//kernel 计算
	err = cudaTransformPoints(threads, d_point_cloud, point_cloud.points.size(), d_m);
		if(err != ::cudaSuccess)return false;

    //把点云数据由device复制到host
	err = cudaMemcpy(point_cloud.points.data(), d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return false;

    //最后释放内存
	err = cudaFree(d_m);
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;


return true;
}


