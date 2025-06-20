// main.cpp (modular version)

#include "camera_utils.h"
#include "feature_tracking.h"
#include "frame_processor.h"
#include "euroc_io.h"
#include "c2p_wrapper.h"
#include <filesystem> 

#include <pybind11/embed.h>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <iostream>
#include <string>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};
    std::string path_to_euroc_data = "/Users/samucerezo/dev/src/Vi-init-cpp/data/MH_02";

    // Dataset containers
    std::vector<double> timu, tcam, tgt;
    std::vector<Eigen::Vector3d> omega, acc, bg_gt;
    std::vector<Eigen::Quaterniond> qgt;
    std::vector<std::string> cam0_image_names;
    Eigen::Matrix4d tbodycam, tbodyimu;
    int starting_frame_;

    // Load EuRoC dataset
    if (!load_euroc_sequence(path_to_euroc_data, timu, omega, acc, tcam, cam0_image_names,
                             tgt, qgt, bg_gt, tbodycam, tbodyimu, starting_frame_)) {
        std::cerr << "Error loading EuRoC dataset." << std::endl;
        return -1;
    }

    // Load camera calibration
    std::string yaml_file = path_to_euroc_data + "/mav0/cam0/sensor.yaml";
    auto [K, distortion_params, image_size] = load_camera_calibration(yaml_file);
    cv::Mat K_cv; cv::eigen2cv(K, K_cv);
    cv::Mat dist_cv(distortion_params);


    // results folder
    std::filesystem::path results_dir = std::filesystem::path("../results");
    if (!std::filesystem::exists(results_dir)) {
        std::filesystem::create_directories(results_dir);
    }
    // Open log file
    std::filesystem::path dataset_path(path_to_euroc_data);
    std::string dataset_name = dataset_path.filename().string();  // e.g., "MH_02"
    std::string output_filename = results_dir / ("results_" + dataset_name + ".csv");

    std::ofstream log_file(output_filename, std::ios::out);
    log_file << "frame,error_aprox,elapsed_aprox_us,error_opt,elapsed_opt_us\n";

    // Frame-by-frame processing
    for (int frame = starting_frame_; frame < cam0_image_names.size() - 1; frame += 5) {
    //for (int frame = starting_frame_; frame < 30; frame += 5) {
        process_frame_pair(frame, cam0_image_names, tcam, timu, omega, 4999936e-9,
                           tgt, qgt, bg_gt, K, K_cv, dist_cv, tbodycam, tbodyimu, log_file);
    }
    return 0;
}
