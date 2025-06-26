#pragma once

#include <Eigen/Dense>
#include <vector>

struct MartinelliEstimate {
    Eigen::Vector3d v0;
    Eigen::Vector3d ba;
    Eigen::Vector3d g;
    std::vector<double> scales;
    int rank;
    int n_unknowns;
};

MartinelliEstimate estimate_state_martinelli_multi(
    const std::vector<std::vector<Eigen::Matrix3d>>& Rks_all,
    const std::vector<std::vector<Eigen::Vector3d>>& accs_all,
    const std::vector<std::vector<double>>& dts_all,
    const std::vector<Eigen::Vector3d>& t_dirs);
