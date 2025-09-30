#pragma once
#include <optional>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

std::optional<Eigen::Matrix3d> compute_Rij_with_c2p(
    const std::string& img_path_i,
    const std::string& img_path_j,
    const Eigen::Matrix3d& K_eigen,
    const cv::Mat& K_cv,
    const cv::Mat& dist_cv);
