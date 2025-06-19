#include "frame_processor.h"
#include "feature_tracking.h"
#include "lie_utils.h"
#include "bg_small_angle.h"
#include "bg_optimization.h"
#include "c2p_wrapper.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <iostream>

void process_frame_pair(int starting_frame,
                        const std::vector<std::string>& cam0_image_names,
                        const std::vector<double>& tcam,
                        const std::vector<double>& timu,
                        const std::vector<Eigen::Vector3d>& omega,
                        double deltat,
                        const std::vector<double>& tgt,
                        const std::vector<Eigen::Quaterniond>& qgt,
                        const std::vector<Eigen::Vector3d>& bg_gt,
                        const Eigen::Matrix3d& K,
                        const cv::Mat& K_cv,
                        const cv::Mat& dist_cv,
                        const Eigen::Matrix4d& tbodycam,
                        const Eigen::Matrix4d& tbodyimu,
                        std::ofstream& log_file) {
    std::cout << "\n[Frame " << starting_frame << "] Processing image pair" << std::endl;

    // Load and track features
    std::string img_path_1 = cam0_image_names[starting_frame];
    std::string img_path_2 = cam0_image_names[starting_frame + 1];
    std::vector<cv::Point2f> old_points, new_points;
    cv::Mat image_1, image_2;

    if (!detect_and_track_features(img_path_1, img_path_2, old_points, new_points, image_1, image_2)) return;
    if (old_points.size() < 8) return;

    // Undistort and normalize
    cv::Mat old_undist, new_undist, mask;
    cv::undistortPoints(old_points, old_undist, K_cv, dist_cv, cv::noArray(), K_cv);
    cv::undistortPoints(new_points, new_undist, K_cv, dist_cv, cv::noArray(), K_cv);

    int num_points = old_undist.rows;
    Eigen::MatrixXd old_h(3, num_points), new_h(3, num_points);
    for (int i = 0; i < num_points; ++i) {
        old_h.col(i) << old_undist.at<cv::Point2f>(i).x, old_undist.at<cv::Point2f>(i).y, 1.0;
        new_h.col(i) << new_undist.at<cv::Point2f>(i).x, new_undist.at<cv::Point2f>(i).y, 1.0;
    }

    Eigen::MatrixXd f0 = (K.inverse() * old_h).colwise().normalized();
    Eigen::MatrixXd f1 = (K.inverse() * new_h).colwise().normalized();

    // Estimate Essential matrix to get inliers
    cv::findEssentialMat(old_undist, new_undist, K_cv, cv::RANSAC, 0.999, 1.0, mask);
    std::vector<Eigen::Vector3d> f0_inliers, f1_inliers;
    for (int i = 0; i < num_points; ++i) {
        if (mask.at<uchar>(i, 0)) {
            f0_inliers.push_back(f0.col(i));
            f1_inliers.push_back(f1.col(i));
        }
    }

    C2PResult c2p_result;
    try {
        c2p_result = run_c2p(f0_inliers, f1_inliers, starting_frame);
    } catch (const std::exception& e) {
        std::cerr << "[Frame " << starting_frame << "] Error in run_c2p: " << e.what() << std::endl;
        return;
    }

    Eigen::Matrix3d R01 = c2p_result.R;

    // Extract IMU segment
    double t1 = tcam[starting_frame];
    double t2 = tcam[starting_frame + 1];
    std::vector<double> imu_times_segment;
    std::vector<Eigen::Vector3d> omega_segment;

    for (size_t i = 0; i < timu.size(); ++i)
        if (timu[i] >= t1 && timu[i] <= t2) {
            imu_times_segment.push_back(timu[i]);
            omega_segment.push_back(omega[i]);
        }

    if (imu_times_segment.size() < 2) return;

    // Preintegration
    std::vector<double> dts;
    for (size_t i = 1; i < imu_times_segment.size(); ++i)
        dts.push_back(imu_times_segment[i] - imu_times_segment[i - 1]);
    omega_segment.resize(omega_segment.size() - 1);

    Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
    std::vector<Eigen::Vector3d> omega_all;
    for (size_t j = 0; j < omega_segment.size(); ++j) {
        Eigen::Vector3d dtheta = omega_segment[j] * dts[j];
        Rpreint *= lie::ExpMap(dtheta);
        omega_all.push_back(dtheta);
    }

    // Small angle bias estimation
    auto t_small_start = std::chrono::high_resolution_clock::now();
    Eigen::Vector3d b_g = bg_small_angle(omega_segment, deltat, Rpreint, R01, tbodycam, tbodyimu);
    auto t_small_end = std::chrono::high_resolution_clock::now();

    // Optimization-based estimation
    std::vector<Eigen::Vector3d> omega_scaled;
    for (const auto& o : omega_all)
        omega_scaled.push_back(o / deltat);

    auto t_opt_start = std::chrono::high_resolution_clock::now();
    Eigen::Vector3d bgopt = bg_optimization(omega_scaled, deltat, R01, Eigen::Vector3d::Zero(), tbodycam, tbodyimu);
    auto t_opt_end = std::chrono::high_resolution_clock::now();

    // Compare with ground truth
    auto it_gt = std::min_element(tgt.begin(), tgt.end(), [t1](double a, double b) {
        return std::abs(a - t1) < std::abs(b - t1);
    });
    size_t idx_gt1 = std::distance(tgt.begin(), it_gt);
    Eigen::Vector3d bg_gt_ = bg_gt[idx_gt1];

    double error_b_g = (bg_gt_ - b_g).norm();
    double error_b_gopt = (bg_gt_ - bgopt).norm();

    std::chrono::duration<double, std::micro> small_elapsed = t_small_end - t_small_start;
    std::chrono::duration<double, std::micro> opt_elapsed = t_opt_end - t_opt_start;

    std::cout << "[Frame " << starting_frame << "] Error bg --> approx: "
              << error_b_g << " t:" << small_elapsed.count()
              << "µs / opt: " << error_b_gopt << " t:" << opt_elapsed.count() << "µs" << std::endl;

    log_file << starting_frame << ","
             << std::fixed << std::setprecision(9) << error_b_g << ","
             << small_elapsed.count() << ","
             << error_b_gopt << ","
             << opt_elapsed.count() << "\n";
}
