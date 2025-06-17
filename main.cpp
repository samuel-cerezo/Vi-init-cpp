// main.cpp
#include "bg_small_angle.h"
#include "lie_utils.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>

int main() {

    // randomizador para omega_all
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.01, 0.005);    // (mean, std)

    // Simular omega_all con 3 mediciones de rotación pequeñas
    std::vector<Eigen::Vector3d> omega_all;
    int N=10;   //numero de muestras
    for(int i=0; i<N ; i++){
        omega_all.emplace_back(noise(generator), noise(generator), noise(generator));
    };
    

    double deltat = 0.01; // 10 ms entre mediciones

    // simular bias real
    Eigen::Vector3d bias_real(0.003, 0.0, 0.0);

    // Preintegrar rotaciones
    Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
    for (const auto& omega_cleaned : omega_all) {
        Eigen::Vector3d omega = omega_cleaned + bias_real;
        Rpreint *= Eigen::AngleAxisd(omega.norm(), omega.normalized()).toRotationMatrix();
    }

    // Simulamos Rij visual
    Eigen::Matrix3d Rij = Eigen::Matrix3d::Identity();
    for(const auto& omega_visual : omega_all){
        Rij *= Eigen::AngleAxisd(omega_visual.norm(), omega_visual.normalized()).toRotationMatrix();
    }

    // Supongamos transformaciones identidad para cuerpo-cámara e IMU
    Eigen::Matrix4d tbodycam = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d tbodyimu = Eigen::Matrix4d::Identity();

    // Calcular el bias con bg_small_angle
    Eigen::Vector3d bg_est = bg_small_angle(omega_all, deltat, Rpreint, Rij, tbodycam, tbodyimu);

    std::cout << "Bias giroscópico estimado (bg):\n" << bg_est.transpose() << std::endl;

    return 0;
}
