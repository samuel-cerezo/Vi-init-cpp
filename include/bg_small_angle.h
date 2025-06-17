// bg_small_angle.h
#pragma once

#include <Eigen/Dense>
#include<vector>

/**
 * Approximate estimation of gyroscope bias using small-angle assumption.
 *
 * @param omega_all 3xN matrix of delta angle measurements (omega * delta_t)
 * @param deltat Scalar time step (not used here, but kept for interface parity)
 * @param Rpreint Preintegrated rotation from IMU data (SO(3) matrix)
 * @param Rij Relative rotation from vision (SO(3) matrix)
 * @param tbodycam 4x4 extrinsic transform from body to camera
 * @param tbodyimu 4x4 extrinsic transform from body to IMU
 * @return Estimated gyroscope bias (3x1 vector)
 */

Eigen::Vector3d bg_small_angle(
    const std::vector<Eigen::Vector3d>& omega_all,
    double deltat,
    const Eigen::Matrix3d& Rpreint,
    const Eigen::Matrix3d& Rij,
    const Eigen::Matrix4d& tbodycam,
    const Eigen::Matrix4d& tbodyimu);

