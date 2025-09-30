#include "utils.h"
#include <optional>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "feature_tracking.h" 
#include <opencv2/opencv.hpp>
#include "c2p_wrapper.h"
#include "track_manager.h"
#include "feature_tracker.h"
#include "camera_utils.h"
#include "frame_processor.h"
#include "euroc_io.h"
#include <filesystem> 
#include <pybind11/embed.h>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <iostream>
#include "lie_utils.h"


std::optional<Eigen::Matrix3d> compute_Rij_with_c2p(
    const std::string& img_path_i,
    const std::string& img_path_j,
    const Eigen::Matrix3d& K_eigen,
    const cv::Mat& K_cv,
    const cv::Mat& dist_cv)
{
    std::vector<cv::Point2f> pts_i, pts_j;
    cv::Mat img_i, img_j;

    if (!detect_and_track_features(img_path_i, img_path_j, pts_i, pts_j, img_i, img_j))
        return std::nullopt;
    if (pts_i.size() < 8) return std::nullopt;

    cv::Mat und_i, und_j, mask;
    cv::undistortPoints(pts_i, und_i, K_cv, dist_cv, cv::noArray(), K_cv);
    cv::undistortPoints(pts_j, und_j, K_cv, dist_cv, cv::noArray(), K_cv);

    int N = und_i.rows;
    Eigen::MatrixXd f0(3, N), f1(3, N);
    Eigen::Matrix3d Kinv = K_eigen.inverse();

    for (int i = 0; i < N; ++i) {
        double x0 = und_i.at<cv::Point2f>(i).x;
        double y0 = und_i.at<cv::Point2f>(i).y;
        double x1 = und_j.at<cv::Point2f>(i).x;
        double y1 = und_j.at<cv::Point2f>(i).y;

        Eigen::Vector3d p0 = Kinv * Eigen::Vector3d(x0, y0, 1.0);
        Eigen::Vector3d p1 = Kinv * Eigen::Vector3d(x1, y1, 1.0);

        f0.col(i) = p0.normalized();
        f1.col(i) = p1.normalized();
    }

    und_i.convertTo(und_i, CV_64F);
    und_j.convertTo(und_j, CV_64F);
    cv::theRNG().state = 42;

    // Recalcular mask con Essential Matrix
    cv::Mat E = cv::findEssentialMat(und_i, und_j, K_cv, cv::RANSAC, 0.999, 1.0, mask);

    std::vector<Eigen::Vector3d> f0_inliers, f1_inliers;
    for (int i = 0; i < N; ++i) {
        double n0 = f0.col(i).norm();
        double n1 = f1.col(i).norm();
        if (mask.at<uchar>(i, 0) && std::isfinite(n0) && std::isfinite(n1) && n0 > 1e-6 && n1 > 1e-6) {
            f0_inliers.push_back(f0.col(i).normalized());
            f1_inliers.push_back(f1.col(i).normalized());
        }
    }

    if (f0_inliers.size() < 8) return std::nullopt;

    C2PResult result = run_c2p(f0_inliers, f1_inliers, 0);
    return result.R;
}
