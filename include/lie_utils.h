// lie_utils.h
#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace lie {

// Devuelve matriz antisimétrica de un vector 3x1
inline Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return m;
}

// Mapa exponencial para SO(3)
inline Eigen::Matrix3d ExpMap(const Eigen::Vector3d& w) {
    double theta = w.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Vector3d axis = w / theta;
    Eigen::Matrix3d wx = skew_symmetric(axis);

    return Eigen::Matrix3d::Identity()
         + wx * std::sin(theta)
         + wx * wx * (1.0 - std::cos(theta));
}

// Mapa logarítmico de SO(3) a R^3
inline Eigen::Vector3d LogMap(const Eigen::Matrix3d& R) {
    if ((R - Eigen::Matrix3d::Identity()).norm() < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    double cos_theta = (R.trace() - 1.0) / 2.0;
    cos_theta = std::min(1.0, std::max(-1.0, cos_theta));
    double theta = std::acos(cos_theta);

    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    
    // evita division por cero
    if (std::abs(sin_theta) < 1e-8) {
        sin_theta = 1e-8;
    }

    Eigen::Matrix3d lnR = (theta / (2.0 * sin_theta)) * (R - R.transpose());
    return Eigen::Vector3d(lnR(2,1), lnR(0,2), lnR(1,0));
}

} // namespace lie
