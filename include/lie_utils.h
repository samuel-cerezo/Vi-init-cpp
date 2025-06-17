// lie_utils.h
#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace lie {

/// Returns the skew-symmetric matrix associated with a 3D vector.
/// This is used in cross product and SO(3) exponential/logarithm maps.
///
/// Input:
/// - v: 3x1 vector
///
/// Output:
/// - 3x3 skew-symmetric matrix
inline Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return m;
}

/// Exponential map from so(3) to SO(3).
/// Converts a rotation vector (axis-angle) into a 3x3 rotation matrix.
///
/// Input:
/// - w: 3x1 rotation vector
///
/// Output:
/// - 3x3 rotation matrix in SO(3)
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

/// Logarithmic map from SO(3) to so(3).
/// Converts a 3x3 rotation matrix into a rotation vector (axis-angle).
///
/// Input:
/// - R: 3x3 rotation matrix in SO(3)
///
/// Output:
/// - 3x1 rotation vector in so(3)
inline Eigen::Vector3d LogMap(const Eigen::Matrix3d& R) {
    // Identity matrix shortcut
    if ((R - Eigen::Matrix3d::Identity()).norm() < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    double cos_theta = (R.trace() - 1.0) / 2.0;
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);
    double theta = std::acos(cos_theta);

    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    if (std::abs(sin_theta) < 1e-8) {
        sin_theta = 1e-8; // avoid division by zero
    }

    Eigen::Matrix3d lnR = (theta / (2.0 * sin_theta)) * (R - R.transpose());
    return Eigen::Vector3d(lnR(2, 1), lnR(0, 2), lnR(1, 0));
}

} // namespace lie
