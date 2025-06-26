#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <fstream>

void process_multi_frame_segment(int i,
                                 const std::vector<std::string>& cam0_image_names,
                                 const std::vector<double>& tcam,
                                 const std::vector<double>& timu,
                                 const std::vector<Eigen::Vector3d>& omega,
                                 const std::vector<Eigen::Vector3d>& acc,
                                 double deltat,
                                 const std::vector<double>& tgt,
                                 const std::vector<Eigen::Quaterniond>& qgt,
                                 const std::vector<Eigen::Vector3d>& bg_gt,
                                 const std::vector<Eigen::Vector3d>& ba_gt,
                                 const std::vector<Eigen::Vector3d>& v_gt,
                                 const std::vector<Eigen::Vector3d>& g_gt,
                                 const std::vector<Eigen::Vector3d>& p_gt,
                                 const Eigen::Matrix3d& K,
                                 const cv::Mat& K_cv,
                                 const cv::Mat& dist_cv,
                                 const Eigen::Matrix4d& tbodycam,
                                 const Eigen::Matrix4d& tbodyimu,
                                 std::ofstream& log_bias_file);
