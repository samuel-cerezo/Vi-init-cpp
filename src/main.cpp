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
    std::vector<Eigen::Vector3d> omega_clean;
    std::vector<Eigen::Vector3d> omega_bias;
    int N=10;   //numero de muestras
    
    // simular bias real
    Eigen::Vector3d bias_real(0.07, 0.0, 0.02);
 
    for(int i=0; i<N ; i++){
        Eigen::Vector3d omega = Eigen::Vector3d(
        noise(generator), 
        noise(generator), 
        noise(generator));
        omega_clean.push_back(omega);
        omega_bias.push_back(omega + bias_real);
    }

    double deltat = 0.01; // 10 ms entre mediciones

 
    // Preintegrar rotaciones
    Eigen::Matrix3d Rpreint = Eigen::Matrix3d::Identity();
    for (const auto& omega : omega_bias) {
        Rpreint *= lie::ExpMap(omega * deltat);
     }

    // Simulamos Rij visual
    Eigen::Matrix3d Rij = Eigen::Matrix3d::Identity();
    for(const auto& omega : omega_clean){
        Rij *= lie::ExpMap(omega * deltat);
    }

    // Supongamos transformaciones identidad para cuerpo-cámara e IMU
    Eigen::Matrix4d tbodycam = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d tbodyimu = Eigen::Matrix4d::Identity();

    // Calcular el bias con bg_small_angle
    Eigen::Vector3d bg_est = bg_small_angle(omega_bias, deltat, Rpreint, Rij, tbodycam, tbodyimu);

    std::cout << "Bias giroscópico estimado (bg):\n" << bg_est.transpose() << std::endl;

    return 0;
}
