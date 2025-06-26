#include "multiframe_processor.h"
#include "feature_tracking.h"
#include "lie_utils.h"
#include "bg_small_angle.h"
#include "bg_optimization.h"
#include "c2p_wrapper.h"
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <estimate_state_martinelli.h>

void process_multi_frame_segment(int starting_frame,
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
                                 std::ofstream& log_bias_file) {

    const int frame_i = starting_frame;
    const int max_num_j = 10;
    const int min_num_j = 4;
    MartinelliEstimate best_estimate;
    bool found = false;

    for (int num_j = min_num_j; num_j <= max_num_j; ++num_j) {
        std::vector<Eigen::Vector3d> t_dirs;
        std::vector<std::vector<Eigen::Vector3d>> omega_segments, acc_segments;
        std::vector<std::vector<double>> dts_segments;
        std::vector<std::vector<Eigen::Matrix3d>> Rks_segments;

        for (int k = 1; k <= num_j; ++k) {
            int frame_j = frame_i + k;  // Frame j_k
            if (frame_j >= cam0_image_names.size()) break;

            // Detecta y estima t_dir usando C2P
            std::string img_path_1 = cam0_image_names[frame_i];
            std::string img_path_2 = cam0_image_names[frame_j];

            std::vector<cv::Point2f> old_points, new_points;
            cv::Mat image_1, image_2;
            if (!detect_and_track_features(img_path_1, img_path_2, old_points, new_points, image_1, image_2)) continue;
            if (old_points.size() < 8) continue;

            cv::Mat old_undist, new_undist, mask;
            cv::undistortPoints(old_points, old_undist, K_cv, dist_cv, cv::noArray(), K_cv);
            cv::undistortPoints(new_points, new_undist, K_cv, dist_cv, cv::noArray(), K_cv);

            cv::Mat E = cv::findEssentialMat(old_undist, new_undist, K_cv, cv::RANSAC, 0.999, 1.0, mask);
            std::vector<Eigen::Vector3d> f0_inliers, f1_inliers;
            for (int i = 0; i < old_undist.rows; ++i) {
                if (mask.at<uchar>(i, 0)) {
                    Eigen::Vector3d pt1(old_undist.at<cv::Point2f>(i).x, old_undist.at<cv::Point2f>(i).y, 1.0);
                    Eigen::Vector3d pt2(new_undist.at<cv::Point2f>(i).x, new_undist.at<cv::Point2f>(i).y, 1.0);
                    f0_inliers.push_back(pt1.normalized());
                    f1_inliers.push_back(pt2.normalized());
                }
            }

            C2PResult c2p_result;
            try {
                c2p_result = run_c2p(f0_inliers, f1_inliers, frame_i);
            } catch (...) { continue; }
            
            Eigen::Matrix3d R01 = c2p_result.R; 
            Eigen::Vector3d t_cam = c2p_result.t.normalized();  //cambiamos de sist de ref. 
            Eigen::Matrix3d R_bc = tbodycam.block<3,3>(0,0);
            Eigen::Vector3d t_imu = R_bc * t_cam;
            t_dirs.push_back(t_imu);

            std::cout << "[DEBUG] t_dir_cam: " << t_cam.transpose() << "  --> t_dir_imu: " << t_imu.transpose() << std::endl;


            //t_dirs.push_back(c2p_result.t.normalized());

            // IMU segment from t_i to t_j
            double t1 = tcam[frame_i];
            double t2 = tcam[frame_j];

            std::vector<double> imu_ts;
            std::vector<Eigen::Vector3d> omega_seg, acc_seg;

            for (size_t i = 0; i < timu.size(); ++i) {
                if (timu[i] >= t1 && timu[i] <= t2) {
                    imu_ts.push_back(timu[i]);
                    omega_seg.push_back(omega[i]);
                    acc_seg.push_back(acc[i]);
                }
            }

            if (imu_ts.size() < 2) continue;

            std::vector<double> dts;
            for (size_t i = 1; i < imu_ts.size(); ++i)
                dts.push_back(imu_ts[i] - imu_ts[i - 1]);

            omega_seg.resize(dts.size());
            acc_seg.resize(dts.size());

            Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
            std::vector<Eigen::Vector3d> omega_all;
            for (size_t j = 0; j < omega_seg.size(); ++j) {
                Eigen::Vector3d dtheta1 = omega_seg[j] * dts[j];
                Rpreint *= lie::ExpMap(dtheta1);
                omega_all.push_back(dtheta1);
            }
            // gyro bias estimation 
            Eigen::Vector3d bg_est = bg_small_angle(omega_all, deltat, Rpreint, R01, tbodycam, tbodyimu);

            std::vector<Eigen::Matrix3d> Rks;
            Rks.push_back(Eigen::Matrix3d::Identity());
            for (size_t i = 0; i < omega_seg.size(); ++i) {
                Eigen::Vector3d dtheta = (omega_seg[i] - bg_est) * dts[i];
                Rks.push_back(Rks.back() * lie::ExpMap(dtheta));
            }
            Rks.pop_back();

            omega_segments.push_back(omega_seg);
            acc_segments.push_back(acc_seg);
            dts_segments.push_back(dts);
            Rks_segments.push_back(Rks);
        }

        if (t_dirs.size() < min_num_j) continue;  

        MartinelliEstimate est = estimate_state_martinelli_multi(
            Rks_segments, acc_segments, dts_segments, t_dirs);

        if (est.rank == est.n_unknowns) {
            best_estimate = est;
            found = true;
            std::cout << "[INFO] Well conditioned system with  " << t_dirs.size() << " pairs ✅\n";
            break;
        } else {
            std::cout << "[INFO] Try with " << t_dirs.size() << " pairs resulted in rank  "
                      << est.rank << " / " << est.n_unknowns << "\n";
        }
    }

    if (!found) {
        std::cerr << "[WARNING] It was not possible to bien a well conditioned system.\n";
        return;
    }

    // Normaliza gravedad
    best_estimate.g.normalize();
    best_estimate.g *= 9.81;

    std::cout << "\n[Frame " << starting_frame << "] === Multi-Martinelli Estimate ===" << std::endl;
    std::cout << "v0: " << best_estimate.v0.transpose() << std::endl;
    std::cout << "ba: " << best_estimate.ba.transpose() << std::endl;
    std::cout << "g:  " << best_estimate.g.transpose() << std::endl;
    std::cout << "||g||: " << best_estimate.g.norm() << "   ángulo con z: " << std::acos(best_estimate.g.normalized().dot(Eigen::Vector3d(0, 0, -1))) * 180 / M_PI << " grados" << std::endl;

    for (int i = 0; i < best_estimate.scales.size(); ++i)
        std::cout << "s" << i << ": " << best_estimate.scales[i] << std::endl;
}
