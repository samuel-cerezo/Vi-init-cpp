// main.cpp
#include "bg_small_angle.h"
#include "lie_utils.h"

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>

int main() {
    // Random generator for synthetic angular velocities (omega)
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.01, 0.005);  // mean=0.01, std=0.005

    // Number of IMU samples
    const int num_samples = 10;

    // True gyroscope bias to be estimated
    Eigen::Vector3d b_g_true(0.07, 0.0, 0.02);

    // Containers for bias-free and biased IMU angular velocities
    std::vector<Eigen::Vector3d> omega_clean;
    std::vector<Eigen::Vector3d> omega_biased;

    // Generate synthetic IMU measurements
    for (int i = 0; i < num_samples; ++i) {
        Eigen::Vector3d omega_noise(noise(generator), noise(generator), noise(generator));
        omega_clean.push_back(omega_noise);
        omega_biased.push_back(omega_noise + b_g_true);
    }

    // Time interval between consecutive IMU samples [s]
    const double delta_t = 0.01;

    // Preintegrated rotation from biased IMU measurements
    Eigen::Matrix3d R_preint = Eigen::Matrix3d::Identity();
    for (const auto& omega : omega_biased) {
        R_preint *= lie::ExpMap(omega * delta_t);
    }

    // Simulated visual rotation using bias-free IMU measurements
    Eigen::Matrix3d R_ij = Eigen::Matrix3d::Identity();
    for (const auto& omega : omega_clean) {
        R_ij *= lie::ExpMap(omega * delta_t);
    }

    // Set identity transforms for body-to-camera and body-to-IMU extrinsics
    Eigen::Matrix4d T_body_cam = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_body_imu = Eigen::Matrix4d::Identity();

    // Estimate gyroscope bias from rotational residuals
    Eigen::Vector3d b_g_est = bg_small_angle(omega_biased, delta_t, R_preint, R_ij, T_body_cam, T_body_imu);

    // Print result
    std::cout << "Estimated gyroscope bias (b_g):\n" << b_g_est.transpose() << std::endl;

    return 0;
}
