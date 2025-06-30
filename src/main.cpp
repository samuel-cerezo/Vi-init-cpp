// main.cpp (modular version)

#include "camera_utils.h"
#include "feature_tracking.h"
#include "frame_processor.h"
#include "multiframe_processor.h"
#include "euroc_io.h"
#include "c2p_wrapper.h"
#include <filesystem> 
#include <pybind11/embed.h>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <estimate_state_martinelli.h>

namespace py = pybind11;

int main(int argc, char** argv) {
    py::scoped_interpreter guard{};

        if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <path_to_dataset>" << std::endl;
        return -1;
    }
    
    std::string path_to_euroc_data = argv[1];


    // Dataset containers
    std::vector<double> timu, tcam, tgt;
    std::vector<Eigen::Vector3d> omega, acc, bg_gt, ba_gt, v_gt, g_gt, p_gt;
    std::vector<Eigen::Quaterniond> qgt;
    std::vector<std::string> cam0_image_names;
    Eigen::Matrix4d tbodycam, tbodyimu;
    int starting_frame_;


    // Load EuRoC dataset
    if (!load_euroc_sequence(path_to_euroc_data, timu, omega, acc, tcam, cam0_image_names,
                             tgt, qgt, bg_gt, ba_gt, v_gt, g_gt, p_gt, tbodycam, tbodyimu, starting_frame_)) {
        std::cerr << "Error loading EuRoC dataset." << std::endl;
        return -1;
    }
    std::cerr << "despues de cargar_dataset..." << std::endl;
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
    std::ofstream log_bias_file(output_filename, std::ios::out);
    log_bias_file << "frame, error_small, elapsed_small_us, error_opt, elapsed_opt_us, error_constVel, elapsed_consVel_us\n";


    //for (int frame = starting_frame_; frame < 30; frame += 5) {
    
    // Frame-by-frame processing --> bg estimation and comparison. The result is saved in a CSV format, in results folder.
    for (int frame = starting_frame_; frame < cam0_image_names.size() - 1; frame += 10) {
        process_frame_pair(frame, cam0_image_names, tcam, timu, omega, acc, 4999936e-9,
                           tgt, qgt, bg_gt, ba_gt, v_gt, g_gt, p_gt, K, K_cv, dist_cv, tbodycam, tbodyimu, log_bias_file);
    }
    

    return 0;
}



// example of use:

//          ./vi_init /Users/samucerezo/dev/src/Vi-init-cpp/data/MH_01