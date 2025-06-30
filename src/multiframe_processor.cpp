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

// Computes scale error percentage between estimated and ground truth displacement
double compute_scale_error(const Eigen::Vector3d& p0_gt,
                           const Eigen::Vector3d& p1_gt,
                           const Eigen::Vector3d& t_dir,
                           double scale_estimated)
{
    double displacement_gt = (p1_gt - p0_gt).norm();
    double displacement_est = scale_estimated * t_dir.norm();

    if (displacement_gt < 1e-6) {
        std::cerr << "[WARNING] GT displacement too small for scale error evaluation." << std::endl;
        return -1.0;
    }

    return std::abs(displacement_est - displacement_gt) / displacement_gt * 100.0;
}


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

    std::vector<Eigen::Vector3d> t_dirs;
    std::vector<std::vector<Eigen::Vector3d>> omega_segments, acc_segments;
    std::vector<std::vector<double>> dts_segments;
    std::vector<std::vector<Eigen::Matrix3d>> Rks_segments;

    for (int num_j = min_num_j; num_j <= max_num_j; ++num_j) {


        for (int k_index = 1; k_index <= num_j; ++k_index) {
            int frame_j = frame_i + k_index;  // Frame j_k
            if (frame_j >= cam0_image_names.size()) break;
            std::cout << "---- trying with " << k_index + 1 << " frames" << std::endl;

            // estimate t_dir using C2P
            std::string img_path_1 = cam0_image_names[frame_i];
            std::string img_path_2 = cam0_image_names[frame_j];

            std::vector<cv::Point2f> old_points, new_points;
            cv::Mat image_1, image_2;
            if (!detect_and_track_features(img_path_1, img_path_2, old_points, new_points, image_1, image_2)) continue;
            if (old_points.size() < 8) continue;

            // Undistort to pixel coordinates using P=K 
            cv::Mat old_undist, new_undist, mask;
            cv::undistortPoints(old_points, old_undist, K_cv, dist_cv, cv::noArray(), K_cv);
            cv::undistortPoints(new_points, new_undist, K_cv, dist_cv, cv::noArray(), K_cv);

            // Transform to Eigen
            int num_points = old_undist.rows;
            Eigen::MatrixXd f0(3, num_points), f1(3, num_points);

            Eigen::Matrix3d K_eigen;
            cv::cv2eigen(K_cv, K_eigen);  

            Eigen::Matrix3d Kinv = K_eigen.inverse();

            for (int i = 0; i < num_points; ++i) {
                double x0 = old_undist.at<cv::Point2f>(i).x;
                double y0 = old_undist.at<cv::Point2f>(i).y;
                double x1 = new_undist.at<cv::Point2f>(i).x;
                double y1 = new_undist.at<cv::Point2f>(i).y;

                Eigen::Vector3d p0 = Kinv * Eigen::Vector3d(x0, y0, 1.0);
                Eigen::Vector3d p1 = Kinv * Eigen::Vector3d(x1, y1, 1.0);

                f0.col(i) = p0.normalized();
                f1.col(i) = p1.normalized();
            }

            old_undist.convertTo(old_undist, CV_64F);
            new_undist.convertTo(new_undist, CV_64F);
            cv::theRNG().state = 42;
            cv::Mat E = cv::findEssentialMat(old_undist, new_undist, K_cv, cv::RANSAC, 0.999, 1.0, mask);
            
            // debug_essential(old_undist, new_undist, mask);

            std::vector<Eigen::Vector3d> f0_inliers, f1_inliers;
            int total_valid = 0, discarded = 0;

            for (int i = 0; i < num_points; ++i) {
                double norm_f0 = f0.col(i).norm();
                double norm_f1 = f1.col(i).norm();

                if (mask.at<uchar>(i, 0) && norm_f0 > 1e-6 && norm_f1 > 1e-6 &&
                    std::isfinite(norm_f0) && std::isfinite(norm_f1)) {
                    
                    f0_inliers.push_back(f0.col(i).normalized());
                    f1_inliers.push_back(f1.col(i).normalized());
                    total_valid++;
                } else {
                    discarded++;
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
            Eigen::Vector3d t_dir = c2p_result.t.normalized();  // direction without scale
            if (!std::isfinite(t_dir.norm()) || t_dir.norm() < 1e-3) {
                std::cerr << "[WARNING] Discarding invalid t_dir at k=" << k_index << std::endl;
                continue;
            }


            //Eigen::Vector3d t_cam = c2p_result.t.normalized();  
            //Eigen::Matrix3d R_bc = tbodycam.block<3,3>(0,0);
            //Eigen::Vector3d t_imu = R_bc * t_cam;
            t_dirs.push_back(t_dir);

            //std::cout << "[DEBUG] t_dir_cam: " << t_cam.transpose() << "  --> t_dir_imu: " << t_imu.transpose() << std::endl;

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

            if (imu_ts.size() < 2) return;
            acc_seg.resize(acc_seg.size() - 1);  

            if (acc_seg.empty()) {
                std::cerr << "acc_segment is empty!" << std::endl;
                return;
            }

            //if (imu_ts.size() < 2) continue;

            std::vector<double> dts;
            for (size_t i = 1; i < imu_ts.size(); ++i)
                dts.push_back(imu_ts[i] - imu_ts[i - 1]);

            omega_seg.resize(omega_seg.size() - 1);

            if (acc_seg.empty() || dts.empty() || omega_seg.size() != dts.size()) {
                std::cerr << "[WARNING] Skipping segment due to IMU mismatch.\n";
                continue;
            }

            Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
            std::vector<Eigen::Vector3d> omega_all;
            for (size_t j = 0; j < omega_seg.size(); ++j) {
                Eigen::Vector3d dtheta = omega_seg[j] * dts[j];
                Rpreint *= lie::ExpMap(dtheta);
                omega_all.push_back(omega_seg[j]);
            }
            std::cout << "R_preint: " << Rpreint << std::endl;

            // gyro bias estimation 
            Eigen::Vector3d bg_est = bg_small_angle(omega_all, deltat, Rpreint, R01, tbodycam, tbodyimu);

            // Compare with ground truth
            auto it_gt = std::min_element(tgt.begin(), tgt.end(), [t1](double a, double b) {
                return std::abs(a - t1) < std::abs(b - t1);
            });
            size_t idx_gt1 = std::distance(tgt.begin(), it_gt);
            Eigen::Vector3d bg_gt_ = bg_gt[idx_gt1];

            double error_b_g = (bg_gt_ - bg_est).norm();
            std::cout << "[k_index "<< k_index << "] " << "bg_error: " << error_b_g << std::endl;

            std::vector<Eigen::Matrix3d> Rks;
            Eigen::Matrix3d Rpreint_corrected = Eigen::Matrix3d::Identity();
            Rks.push_back(Eigen::Matrix3d::Identity());
            for (size_t i = 0; i < omega_seg.size(); ++i) {
                Eigen::Vector3d dtheta1 = (omega_seg[i] - bg_est) * dts[i];
                Rpreint_corrected *= lie::ExpMap(dtheta1);
                Rks.push_back(Rks.back()*=lie::ExpMap(dtheta1));
            }
            Rks.pop_back();
            std::cout << "R_corregida: " << Rpreint_corrected << std::endl;

            omega_segments.push_back(omega_seg);
            acc_segments.push_back(acc_seg);
            dts_segments.push_back(dts);
            Rks_segments.push_back(Rks);
        }

        
        MartinelliEstimate est = estimate_state_martinelli_multi(
            Rks_segments, acc_segments, dts_segments, t_dirs);

        // Normaliza gravedad
        best_estimate.g.normalize();
        best_estimate.g *= 9.81;
        const Eigen::Vector3d& v0_gt = v_gt[frame_i];

        std::cout << "\n[Frame " << starting_frame << "] === Full initialization ===" << std::endl;
        std::cout << "v0: " << best_estimate.v0.transpose() << std::endl;
        std::cout << "v0_gt: " <<  v0_gt.transpose() << std::endl;
        std::cout << "ba: " << best_estimate.ba.transpose() << std::endl;
        std::cout << "g:  " << best_estimate.g.transpose() << std::endl;

    }

    

    /*
    // === Compute and display scale errors ===
    for (int i = 0; i < best_estimate.scales.size(); ++i) {
        int frame_j = frame_i + i + 1;

        if (frame_j >= p_gt.size()) continue;

        const Eigen::Vector3d& p0_gt = p_gt[frame_i];
        const Eigen::Vector3d& p1_gt = p_gt[frame_j];
        const Eigen::Vector3d& t_dir = t_dirs[i];
        double s_est = best_estimate.scales[i];

        std::cout << "p0_gt: " << p0_gt.transpose() << " p1_gt: " << p1_gt.transpose() << " t_estim: " << t_dir.transpose() << std::endl;


        double scale_err = compute_scale_error(p0_gt, p1_gt, t_dir, s_est);

        std::cout << "s" << i << ": " << s_est
                << "  (scale error: " << scale_err << "%)" << std::endl;
    }

    */



}
