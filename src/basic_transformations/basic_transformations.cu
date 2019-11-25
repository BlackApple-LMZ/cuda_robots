#include "basic_transformations/basic_transformations.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>


__global__ void kernel_cudaWarmUpGPU()
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	ind = ind + 1;
    //printf("Device Name : %d.\n", ind); //每个warp按照顺序打印 512 对应了16个warp 这16个warp打印顺序是不确定的
}

cudaError_t cudaWarmUpGPU()
{
	kernel_cudaWarmUpGPU<<<1,1>>>();
	cudaDeviceSynchronize(); //Wait for compute device to finish 等所有线程都运行结束
	return cudaGetLastError(); 
}

__global__ void kernel_cudaTransformPoints(pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;




	//因为分配的线程数量大于实际的点云数量 这里的判断是防止溢出
	if(ind<number_of_points)
	{
		float vSrcVector[3] = {d_point_cloud[ind].x, d_point_cloud[ind].y, d_point_cloud[ind].z};
		float vOut[3];

		//对每个点计算转换后的坐标
		vOut[0]=d_matrix[0]*vSrcVector[0]+d_matrix[4]*vSrcVector[1]+d_matrix[8]*vSrcVector[2]+d_matrix[12];
   	 	vOut[1]=d_matrix[1]*vSrcVector[0]+d_matrix[5]*vSrcVector[1]+d_matrix[9]*vSrcVector[2]+d_matrix[13];
    	vOut[2]=d_matrix[2]*vSrcVector[0]+d_matrix[6]*vSrcVector[1]+d_matrix[10]*vSrcVector[2]+d_matrix[14];

		d_point_cloud[ind].x = vOut[0];
		d_point_cloud[ind].y = vOut[1];
		d_point_cloud[ind].z = vOut[2];
	}
}

cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix)
{
    //分配足够的threads 计算点云变换
    //kernel前两个参数的类型 决定了 grid 和 block的维度 这里都是1维的 也可以是dim3 维的
	kernel_cudaTransformPoints<<<number_of_points/threads+1,threads>>>
		(d_point_cloud, number_of_points, d_matrix);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

__global__ void kernel_cudaRemovePointsInsideSphere
	(pcl::PointXYZ *d_point_cloud, bool *d_markers, int number_of_points, float sphere_radius)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<number_of_points)
	{
		float x = d_point_cloud[ind].x;
		float y = d_point_cloud[ind].y;
		float z = d_point_cloud[ind].z;
	
		float distance = (x*x + y*y + z*z);
	
		if(distance < sphere_radius * sphere_radius)
		{
			d_markers[ind] = false;
		}else
		{
			d_markers[ind] = true;
		}
	}
}

cudaError_t cudaRemovePointsInsideSphere(int threads, pcl::PointXYZ *d_point_cloud, 
		bool *d_markers, int number_of_points, float sphere_radius)
{
    //    dim3 dimGrid(number_of_points/threads+1,1,1);
    //    dim3 dimBlock(threads,1,1);
	kernel_cudaRemovePointsInsideSphere<<<number_of_points/threads+1,threads>>>
		(d_point_cloud,	d_markers, number_of_points, sphere_radius);
		
	cudaDeviceSynchronize();
	return cudaGetLastError();
}






