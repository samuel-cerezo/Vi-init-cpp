// euroc_io.cpp
#include "euroc_io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <yaml-cpp/yaml.h>


static bool load_TBS(const std::string& yaml_file, Eigen::Matrix4d& T) {
    YAML::Node config = YAML::LoadFile(yaml_file);
    if (!config["T_BS"]) return false;

    auto data = config["T_BS"]["data"];
    if (!data || data.size() != 16) return false;

    for (int i = 0; i < 16; ++i) {
        T(i / 4, i % 4) = data[i].as<double>();
    }
    return true;
}

static bool load_csv_imu(const std::string& filename,
                         std::vector<double>& timestamps,
                         std::vector<Eigen::Vector3d>& omega,
                         std::vector<Eigen::Vector3d>& acc)
{
    std::ifstream file(filename); std::string line;
    if (!file.is_open()) return false;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v;
        std::vector<double> row;
        while (std::getline(ss, v, ',')) row.push_back(std::stod(v));
        if (row.size() == 7) {
            timestamps.push_back(row[0] * 1e-9);
            omega.emplace_back(row[1], row[2], row[3]);
            acc.emplace_back(row[4], row[5], row[6]);
        }
    }
    return true;
}

static bool load_csv_cam(const std::string& filename, const std::string& base_path,
                         std::vector<double>& timestamps,
                         std::vector<std::string>& filenames)
{
    std::ifstream file(filename); std::string line;
    if (!file.is_open()) return false;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string ts, fn;
        std::getline(ss, ts, ','); std::getline(ss, fn);
        timestamps.push_back(std::stod(ts) * 1e-9);
        filenames.push_back(base_path + "/" + fn);
    }
    return true;
}

static bool load_csv_gt(const std::string& filename,
                        std::vector<double>& timestamps,
                        std::vector<Eigen::Quaterniond>& quats,
                        std::vector<Eigen::Vector3d>& biases)
{
    std::ifstream file(filename); std::string line;
    if (!file.is_open()) return false;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line); std::string v;
        std::vector<double> row;
        while (std::getline(ss, v, ',')) row.push_back(std::stod(v));
        if (row.size() >= 15) {
            timestamps.push_back(row[0] * 1e-9);
            quats.emplace_back(row[7], row[4], row[5], row[6]);  // qw, qx, qy, qz
            biases.emplace_back(row[11], row[12], row[13]);
        }
    }
    return true;
}

bool load_euroc_sequence(const std::string& euroc_path,
                         std::vector<double>& timu,
                         std::vector<Eigen::Vector3d>& omega,
                         std::vector<Eigen::Vector3d>& acc,
                         std::vector<double>& tcam,
                         std::vector<std::string>& cam0_image_names,
                         std::vector<double>& tgt,
                         std::vector<Eigen::Quaterniond>& qgt,
                         std::vector<Eigen::Vector3d>& bg_gt,
                         Eigen::Matrix4d& tbodycam,
                         Eigen::Matrix4d& tbodyimu)
{
    std::string imu_file = euroc_path + "/mav0/imu0/data.csv";
    std::string cam_csv = euroc_path + "/mav0/cam0/data.csv";
    std::string cam_img_dir = euroc_path + "/mav0/cam0/data";
    std::string gt_file = euroc_path + "/mav0/state_groundtruth_estimate0/data.csv";
    std::string cam_yaml = euroc_path + "/mav0/cam0/sensor.yaml";
    std::string imu_yaml = euroc_path + "/mav0/imu0/sensor.yaml";

    return load_csv_imu(imu_file, timu, omega, acc)
        && load_csv_cam(cam_csv, cam_img_dir, tcam, cam0_image_names)
        && load_csv_gt(gt_file, tgt, qgt, bg_gt)
        && load_TBS(cam_yaml, tbodycam)
        && load_TBS(imu_yaml, tbodyimu);
}
