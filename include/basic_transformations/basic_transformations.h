#ifndef _BASIC_TRANSFORMATIONS_H_
#define _BASIC_TRANSFORMATIONS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>

cudaError_t cudaWarmUpGPU();
cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix);
cudaError_t cudaRemovePointsInsideSphere(int threads, pcl::PointXYZ *d_point_cloud, bool *d_markers, int number_of_points, float sphere_radius);


#endif
