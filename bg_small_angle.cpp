// bg_small_angle.cpp
#include "bg_small_angle.h"
#include "lie_utils.h"
#include <Eigen/Dense>

Eigen::Vector3d bg_small_angle(
    const std::vector<Eigen::Vector3d>& omega_all,
    double deltat,
    const Eigen::Matrix3d& Rpreint,
    const Eigen::Matrix3d& Rij,
    const Eigen::Matrix4d& tbodycam,
    const Eigen::Matrix4d& tbodyimu)
{
    // Extracción de rotaciones
    Eigen::Matrix3d R_imu = tbodyimu.topLeftCorner<3,3>();
    Eigen::Matrix3d R_cam = tbodycam.topLeftCorner<3,3>();

    // Composición de rotaciones
    Eigen::Matrix3d R_integrated = R_imu * Rpreint * R_imu.transpose();
    Eigen::Matrix3d R_measured = R_cam * Rij * R_cam.transpose();

    // Error rotacional
    Eigen::Matrix3d R_error = R_integrated.transpose() * R_measured;
    Eigen::Vector3d log_rot = lie::LogMap(R_error);

    // Estimación de bias
    int n = static_cast<int>(omega_all.size());
    Eigen::Vector3d bg = - (1.0 / (n * deltat)) * (R_imu.transpose() * log_rot);

    return bg;
}
