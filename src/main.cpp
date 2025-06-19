// main.cpp
#include "bg_small_angle.h"
#include "bg_optimization.h"
#include "lie_utils.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include "euroc_io.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pybind11/embed.h>
#include "c2p_wrapper.h"
#include <pybind11/eigen.h>  // <== NECESARIO para usar Eigen::MatrixXd con pybind11
#include <chrono>
#include <fstream>
#include <iomanip>



namespace py = pybind11;

// Declaración del wrapper
C2PResult run_c2p(const std::vector<Eigen::Vector3d>& f0_inliers,
             const std::vector<Eigen::Vector3d>& f1_inliers,
             int frame_idx);

bool detect_and_track_features(const std::string& img1_path, const std::string& img2_path,
                               std::vector<cv::Point2f>& old_pts,
                               std::vector<cv::Point2f>& new_pts,
                               cv::Mat& img1, cv::Mat& img2) 
{
    img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) return false;

    std::vector<cv::Point2f> points1;
    cv::goodFeaturesToTrack(img1, points1, 300, 0.01, 10);
    if (points1.empty()) return false;

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(img1, img2, points1, new_pts, status, err);

    // Filtrar puntos válidos
    old_pts.clear();
    std::vector<cv::Point2f> filtered_new;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            old_pts.push_back(points1[i]);
            filtered_new.push_back(new_pts[i]);
        }
    }
    new_pts = filtered_new;

    return !old_pts.empty();
}



struct CameraCalibration {
    Eigen::Matrix3d K;
    std::vector<double> distortion_params;
    std::pair<int, int> image_size;
};

CameraCalibration load_camera_calibration(const std::string& yaml_file) {
    YAML::Node config = YAML::LoadFile(yaml_file);
    CameraCalibration calib;

    auto intrinsics = config["intrinsics"];
    if (!intrinsics || intrinsics.size() < 4)
        throw std::runtime_error("Missing or invalid intrinsics");

    double fx = intrinsics[0].as<double>();
    double fy = intrinsics[1].as<double>();
    double cx = intrinsics[2].as<double>();
    double cy = intrinsics[3].as<double>();

    calib.K << fx, 0,  cx,
                0, fy, cy,
                0,  0,  1;

    auto distortion = config["distortion_coefficients"];
    for (size_t i = 0; i < distortion.size(); ++i) {
        calib.distortion_params.push_back(distortion[i].as<double>());
    }

    auto resolution = config["resolution"];
    if (resolution && resolution.size() == 2) {
        calib.image_size = { resolution[0].as<int>(), resolution[1].as<int>() };
    }

    return calib;
}



int main() {
    
    py::scoped_interpreter guard{};  // 

    std::string path_to_euroc_data = "/Users/samucerezo/dev/src/Vi-init-cpp/data/MH_02_easy";

    std::vector<double> timu, tcam, tgt;
    std::vector<Eigen::Vector3d> omega, acc, bg_gt;
    std::vector<Eigen::Quaterniond> qgt;
    std::vector<std::string> cam0_image_names;
    Eigen::Matrix4d tbodycam, tbodyimu;
    int starting_frame_;  

    if (!load_euroc_sequence(path_to_euroc_data,
                             timu, omega, acc, tcam, cam0_image_names,
                             tgt, qgt, bg_gt, tbodycam, tbodyimu, starting_frame_)) {
        std::cout << "Primera imagen: " << cam0_image_names[0] << std::endl;
        std::cerr << "Error loading EuRoC dataset." << std::endl;
        return -1;
    }
    

    std::string path_to_yaml_cam = path_to_euroc_data + "/mav0/cam0/sensor.yaml";

    auto [K, distortion_params, image_size] = load_camera_calibration(path_to_yaml_cam);
    Eigen::Matrix3d Kinv = K.inverse();
    cv::Mat K_cv;
    cv::eigen2cv(K, K_cv); 
    cv::Mat dist_cv(distortion_params);  // si ya es cv::Mat OK, si no, convertí



    
    //std::cout << "Camera intrinsic matrix K:\n" << K << std::endl;
    //std::cout << "Inverse K (Kinv):\n" << K.inverse() << std::endl;
    //std::cout << "Distortion parameters:\n";
    //for (size_t i = 0; i < distortion_params.size(); ++i) {
    //    std::cout << "  param[" << i << "] = " << distortion_params[i] << std::endl;
    //}
    //std::cout << "Image size: " << image_size.first << " x " << image_size.second << std::endl;

    int num_frames = 1;
    double deltat = 4999936 * 1e-9; // segundos

    // Initialize evaluation logs
    std::vector<Eigen::Vector3d> b_g_all;           // Estimated gyroscope biases
    std::vector<Eigen::Vector3d> b_gopt_all;        // Optimized gyroscope biases
    std::vector<double> error_b_g_all;              // Errors in initial estimation
    std::vector<double> error_b_gopt_all;           // Errors in optimized estimation
    std::vector<int> frame_list;                    // List of frame indices

    
        auto print_matrix = [](const Eigen::Matrix3d& M, const std::string& name) {
        std::cout << name << " =\n";
        for (int i = 0; i < 3; ++i)
            std::cout << "  " << M(i, 0) << "   " << M(i, 1) << "   " << M(i, 2) << std::endl;
    };

    Eigen::Matrix3d R_cam = tbodycam.topLeftCorner<3,3>();
    Eigen::Matrix3d R_imu = tbodyimu.topLeftCorner<3,3>();
    Eigen::Matrix3d R_cam_imu = R_cam * R_imu.transpose();

    print_matrix(R_cam, "R_cam");
    print_matrix(R_imu, "R_imu");
    print_matrix(R_cam_imu, "R_cam_imu");


    /// los resultados se guardaran aqui:
    std::ofstream log_file("results.csv", std::ios::out);
    log_file << "frame,error_aprox,elapsed_aprox_us,error_opt,elapsed_opt_us\n";


    for (int starting_frame = starting_frame_; starting_frame < static_cast<int>(cam0_image_names.size()) - 1; starting_frame += 5) {

        std::cout << "\n[Frame " << starting_frame << "] Procesando par de imágenes" << std::endl;

        std::string img_path_1 = cam0_image_names[starting_frame];
        std::string img_path_2 = cam0_image_names[starting_frame + 1];

        std::vector<cv::Point2f> old_points, new_points;
        cv::Mat image_1, image_2;

        if (!detect_and_track_features(img_path_1, img_path_2, old_points, new_points, image_1, image_2)) {
            std::cout << "[Frame " << starting_frame << "] Error al detectar/tracking o pocos puntos" << std::endl;
            continue;
        }

        if (old_points.size() < 8) {
            std::cout << "[Frame " << starting_frame << "] Pocos puntos detectados" << std::endl;
            continue;
        }

        // Undistort points
        cv::Mat E, mask;

        cv::Mat old_undist, new_undist;
        cv::undistortPoints(old_points, old_undist, K_cv, dist_cv, cv::noArray(), K_cv);
        cv::undistortPoints(new_points, new_undist, K_cv, dist_cv, cv::noArray(), K_cv);
        E = cv::findEssentialMat(old_undist, new_undist, K_cv, cv::RANSAC, 0.999, 1.0, mask);

        // Assumimos que old_undist y new_undist son cv::Mat Nx2 y Kinv es Eigen::Matrix3d

        int num_points = old_undist.rows;
        Eigen::MatrixXd old_h(3, num_points);
        Eigen::MatrixXd new_h(3, num_points);

        for (int i = 0; i < num_points; ++i) {
            old_h.col(i) << old_undist.at<cv::Point2f>(i).x, old_undist.at<cv::Point2f>(i).y, 1.0;
            new_h.col(i) << new_undist.at<cv::Point2f>(i).x, new_undist.at<cv::Point2f>(i).y, 1.0;
        }

        // convierte puntos homogéneos proyectados en rayos normalizados
        Eigen::MatrixXd f0_cam = Kinv * old_h;
        Eigen::MatrixXd f1_cam = Kinv * new_h;

        Eigen::MatrixXd f0(3, num_points);
        Eigen::MatrixXd f1(3, num_points);

        for (int i = 0; i < num_points; ++i) {
            f0.col(i) = f0_cam.col(i).normalized();
            f1.col(i) = f1_cam.col(i).normalized();
        }


        E = cv::findEssentialMat(old_undist, new_undist, K_cv, cv::RANSAC, 0.999, 1.0, mask);

        // Extrae los puntos inlier
        std::vector<Eigen::Vector3d> f0_inliers, f1_inliers;


        for (int i = 0; i < num_points; ++i) {
            if (mask.at<uchar>(i, 0)) {
                f0_inliers.push_back(f0.col(i));
                f1_inliers.push_back(f1.col(i));
            }
        }

        /* Eigen::MatrixXd f0_np(3, f0_inliers.size());
        Eigen::MatrixXd f1_np(3, f1_inliers.size());

        for (size_t i = 0; i < f0_inliers.size(); ++i) {
            f0_np.col(i) = f0_inliers[i];
            f1_np.col(i) = f1_inliers[i];
        }
        */

        C2PResult c2p_result;



        try {
            c2p_result = run_c2p(f0_inliers, f1_inliers, starting_frame);
        } catch (const std::exception& e) {
            std::cerr << "[Frame " << starting_frame << "] Error en run_c2p: " << e.what() << std::endl;
            continue;
        }

        Eigen::Matrix3d R01 = c2p_result.R;

        // filtrado de mediciones IMU en el intervalo [t1, t2]
        double t1 = tcam[starting_frame];
        double t2 = tcam[starting_frame + 1];

        std::vector<double> imu_times_segment;
        std::vector<Eigen::Vector3d> omega_segment;

        for (size_t i = 0; i < timu.size(); ++i) {
            if (timu[i] >= t1 && timu[i] <= t2) {
                imu_times_segment.push_back(timu[i]);
                omega_segment.push_back(omega[i]);
            }
        }

        if (imu_times_segment.size() < 2) {
            std::cerr << "[Frame " << starting_frame << "] [ADVERTENCIA] Muy pocos datos IMU entre imágenes" << std::endl;
            continue;
        }


        // calculo de dts y recorte de omega_segment
        std::vector<double> dts;
        for (size_t i = 1; i < imu_times_segment.size(); ++i) {
            dts.push_back(imu_times_segment[i] - imu_times_segment[i - 1]);
        }
        omega_segment.resize(omega_segment.size() - 1);  // Para alinear con dts


        // Rot preintegration
        Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
        std::vector<Eigen::Vector3d> omega_all;

        for (size_t j = 0; j < omega_segment.size(); ++j) {
            Eigen::Vector3d delta_theta = omega_segment[j] * dts[j];
            Rpreint = Rpreint * lie::ExpMap(delta_theta);  
            omega_all.push_back(delta_theta);
        }
        std::cout << "Rpreint = \n" << Rpreint << std::endl;

        //bias estimation
        auto t_small_start = std::chrono::high_resolution_clock::now();
        Eigen::Vector3d b_g = bg_small_angle(omega_segment, deltat, Rpreint, R01, tbodycam, tbodyimu);
        auto t_small_end = std::chrono::high_resolution_clock::now();

        Eigen::Vector3d bg0 = Eigen::Vector3d::Zero();
        
        //////
        std::vector<Eigen::Vector3d> omega_scaled;
        omega_scaled.reserve(omega_all.size());
        for (const auto& omega : omega_all) {
            omega_scaled.push_back((1.0 / deltat) * omega);
        }

        auto t_opt_start = std::chrono::high_resolution_clock::now();
        Eigen::Vector3d bgopt = bg_optimization(omega_scaled, deltat, R01, bg0, tbodycam, tbodyimu);
        auto t_opt_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::micro> small_elapsed = t_small_end - t_small_start;
        std::chrono::duration<double, std::micro> opt_elapsed = t_opt_end - t_opt_start;
        
        // gt comparison
        auto it_gt = std::min_element(tgt.begin(), tgt.end(),
            [t1](double a, double b) { return std::abs(a - t1) < std::abs(b - t1); });
        size_t idx_gt1 = std::distance(tgt.begin(), it_gt);

        Eigen::Quaterniond q_gt = qgt[idx_gt1];
        Eigen::Vector3d bg_gt_ = bg_gt[idx_gt1];

        double error_b_g = (bg_gt_ - b_g).norm();
        double error_b_gopt = (bg_gt_ - bgopt).norm();

        frame_list.push_back(starting_frame);
        error_b_g_all.push_back(error_b_g);
        b_g_all.push_back(b_g);
        error_b_gopt_all.push_back(error_b_gopt);
        b_gopt_all.push_back(bgopt);

        std::cout << "[Frame " << starting_frame << "] Error bg -->      aprox: "
                << error_b_g << " t:" << small_elapsed.count() << "µs"" / opt: " << error_b_gopt << " t:" << opt_elapsed.count() << "µs" << std::endl;

        // Guardar en CSV
        log_file << starting_frame << ","
                << std::fixed << std::setprecision(9) << error_b_g << ","
                << small_elapsed.count() << ","
                << error_b_gopt << ","
                << opt_elapsed.count() << "\n";
    }

    return 0;
}
