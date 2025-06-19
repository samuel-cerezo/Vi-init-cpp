// bg_small_angle.cpp
#include "bg_small_angle.h"
#include "lie_utils.h"
#include <iostream>
#include <Eigen/Dense>

// Estimate gyroscope bias using small-angle approximation.
// Inputs:
// - omega_all: sequence of biased angular velocity measurements (in body frame)
// - deltat: sampling time interval [s]
// - Rpreint: rotation from IMU preintegration
// - Rij: relative rotation from vision (camera)
// - tbodycam: 4x4 extrinsic transform from body to camera
// - tbodyimu: 4x4 extrinsic transform from body to IMU
//
// Output:
// - Estimated gyroscope bias vector
Eigen::Vector3d bg_small_angle(
    const std::vector<Eigen::Vector3d>& omega_all,
    double deltat,
    const Eigen::Matrix3d& Rpreint,
    const Eigen::Matrix3d& R_ij,
    const Eigen::Matrix4d& T_body_cam,
    const Eigen::Matrix4d& T_body_imu)
{
    // Extract 3x3 rotation matrices from extrinsics
    Eigen::Matrix3d R_imu = T_body_imu.topLeftCorner<3, 3>();
    Eigen::Matrix3d R_cam = T_body_cam.topLeftCorner<3, 3>();

    // Compose integrated and measured rotation in the same reference frame
    Eigen::Matrix3d R_integrated = R_imu * Rpreint * R_imu.transpose();
    Eigen::Matrix3d R_measured = R_cam * R_ij * R_cam.transpose();

    // Compute the rotational error
    Eigen::Matrix3d R_error = R_integrated.transpose() * R_measured;

    // Map rotation error to so(3)
    Eigen::Vector3d log_rot = lie::LogMapRobust(R_error);
    

    // Bias estimation using small-angle model
    const int n = static_cast<int>(omega_all.size());
    
    std::cout << "n: " << n << ", deltat: " << deltat << std::endl;
    std::cout << "log_rot:\n" << log_rot.transpose() << std::endl;
    std::cout << "R_imu.transpose():\n" << R_imu.transpose() << std::endl;
    Eigen::Vector3d rot_scaled = R_imu.transpose() * log_rot;
    std::cout << "R_imu^T * log_rot = " << rot_scaled.transpose() << std::endl;

    Eigen::Vector3d b_g = - (1.0 / (n * deltat)) * (R_imu.transpose() * log_rot);
    std::cout << "log_rot (C++) = " << log_rot.transpose() << std::endl;

    return b_g;
}
