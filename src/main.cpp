// main.cpp
#include "bg_small_angle.h"
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

Eigen::Matrix4d load_TBS_from_yaml(const std::string& yaml_file) {
    YAML::Node config = YAML::LoadFile(yaml_file);

    if (!config["T_BS"]) {
        throw std::runtime_error("T_BS not found in: " + yaml_file);
    }

    auto data = config["T_BS"]["data"];
    if (!data || data.size() != 16) {
        throw std::runtime_error("Invalid T_BS format in: " + yaml_file);
    }

    Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 16; ++i) {
        T(i / 4, i % 4) = data[i].as<double>();
    }

    return T;
}

bool load_gt_csv(const std::string& filename,
                 std::vector<double>& timestamps,
                 std::vector<Eigen::Quaterniond>& quats,
                 std::vector<Eigen::Vector3d>& gyro_biases) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening GT file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string val;
        while (std::getline(ss, val, ',')) {
            row.push_back(std::stod(val));
        }

        if (row.size() < 15) continue;

        timestamps.push_back(row[0] * 1e-9);
        quats.emplace_back(row[7], row[4], row[5], row[6]);  // qw, qx, qy, qz
        gyro_biases.emplace_back(row[11], row[12], row[13]); // bgx, bgy, bgz
    }

    return true;
}




bool load_imu_csv(const std::string& filename,
                std::vector<double>& timestamps,
                std::vector<Eigen::Vector3d>& omega,
                std::vector<Eigen::Vector3d>& acc)
{
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error opening: " << filename << std::endl;
        return false;
    }

    std::string line;

    //header
    std::getline(file, line);

    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while(std::getline(ss, value, ',')){
            try
            {
                row.push_back(std::stod(value));
            }
            catch(const std::invalid_argument&)
            {
                continue;   //skip invalid values
            }
        }

        if (row.size() != 7) continue;

        timestamps.push_back(row[0] * 1e-9);    //time in sec
        omega.emplace_back(row[1], row[2], row[3]);
        acc.emplace_back(row[4], row[5], row[6]);
    }

    return true;
}





bool load_cam_csv(const std::string& filename, 
                    std::vector<double>& timestamps,
                    std::vector<std::string>& image_filenames,
                    const std::string& image_base_path)
{
    std::ifstream file(filename);
    if(!file.is_open()){
       std::cerr << "Error opening file: " << filename << std::endl;
       return false;
    }

    std::string line;
    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string value;
        std::getline(ss, value, ',');
        double ts = std::stod(value);
        std::getline(ss, value);
        std::string filename = image_base_path + "/" + value;
        timestamps.push_back(ts * 1e-9);
        image_filenames.push_back(filename);
    }

    return true;
}


int main() {

    std::string path_to_euroc_data = "/Users/samucerezo/dev/src/Vi-init-cpp/data/MH_02_easy";

    std::vector<double> timu, tcam, tgt;
    std::vector<Eigen::Vector3d> omega, acc, bg_gt;
    std::vector<Eigen::Quaterniond> qgt;
    std::vector<std::string> cam0_image_names;
    Eigen::Matrix4d tbodycam, tbodyimu;

    if (!load_euroc_sequence(path_to_euroc_data,
                             timu, omega, acc, tcam, cam0_image_names,
                             tgt, qgt, bg_gt, tbodycam, tbodyimu)) {
        std::cerr << "Error loading EuRoC dataset." << std::endl;
        return -1;
    }

    int k=20;   //first frame

    if (k >= tcam.size() - 1) {
        std::cerr << "Índex out of range" << std::endl;
        return -1;
    }

    double t0 = tcam[k];
    double t1 = tcam[k + 1];

    std::vector<Eigen::Vector3d> omega_segment;
    double deltat = 0.0;
    int count = 0;

    for (size_t i = 0; i < timu.size() - 1; ++i) {
        if (timu[i] >= t0 && timu[i] <= t1) {
            omega_segment.push_back(omega[i]);
            deltat += (timu[i+1] - timu[i]);
            ++count;
        }
    }
    if (count == 0) {
        std::cerr << "No se encontraron medidas IMU entre los tiempos indicados" << std::endl;
        return -1;
    }
    deltat /= count;
    
    Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
    for (const auto& w : omega_segment) {
        Rpreint *= lie::ExpMap(w * deltat);
    }


    // using gt for Rij between tgt[k] y tgt[k+1]
    if (k >= tgt.size() - 1) {
        std::cerr << "Ground-truth fuera de rango" << std::endl;
        return -1;
    }
    
    Eigen::Quaterniond q0 = qgt[k].normalized();
    Eigen::Quaterniond q1 = qgt[k+1].normalized();

    Eigen::Matrix3d R0 = q0.normalized().toRotationMatrix();
    Eigen::Matrix3d R1 = q1.normalized().toRotationMatrix();
    Eigen::Matrix3d Rij = R0.transpose() * R1;

    Eigen::Vector3d bg_est = bg_small_angle(omega_segment, deltat, Rpreint, Rij, tbodycam, tbodyimu);
    std::cout << "Bias estimated:\n" << bg_est.transpose() << std::endl;
    return 0;
}
