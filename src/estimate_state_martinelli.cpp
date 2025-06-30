#include "estimate_state_martinelli.h"
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <iostream>
#include <numeric>


MartinelliEstimate estimate_state_martinelli_multi(
    const std::vector<std::vector<Eigen::Matrix3d>>& Rks_all,
    const std::vector<std::vector<Eigen::Vector3d>>& accs_all,
    const std::vector<std::vector<double>>& dts_all,
    const std::vector<Eigen::Vector3d>& t_dirs)
{
    const int num_pairs = Rks_all.size();
    assert(num_pairs == accs_all.size());
    assert(num_pairs == dts_all.size());
    assert(num_pairs == t_dirs.size());

    const int n_unknowns = 3 + 3 + 3 + num_pairs; // v0, ba, g, s_i
    const int n_rows = 3 * num_pairs;

    Eigen::MatrixXd A(n_rows, n_unknowns);
    Eigen::VectorXd b(n_rows);

    for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx)
    {
        const auto& Rks = Rks_all[pair_idx];
        const auto& accs = accs_all[pair_idx];
        const auto& dts = dts_all[pair_idx];
        const Eigen::Vector3d& t_dir = t_dirs[pair_idx];

        const int n = Rks.size();
        assert(n == accs.size());
        assert(n == dts.size());

        double delta_T = 0.0;
        double g_coeff_scalar = 0.0;
        Eigen::Matrix3d A_ba = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b_term = Eigen::Vector3d::Zero();

        // --- Acumuladores internos ---
        for (int k = 0; k < n; ++k)
        {
            double dt_k = dts[k];
            delta_T += dt_k;

            // Para bias del acelerómetro
            A_ba += Rks[k] * dt_k * dt_k;

            // Para el lado derecho
            b_term += Rks[k] * accs[k] * dt_k * dt_k;

            // Para gravedad (t_i - t_l)*(t_j - t_l) ≈ dt_l * dt_k
            for (int l = 0; l <= k; ++l)
                g_coeff_scalar += dts[l] * dt_k;
        }

        const int row_base = 3 * pair_idx;

        // Llenado de la matriz A
        A.block<3, 1>(row_base, pair_idx) = -t_dir;
        A.block<3, 3>(row_base, num_pairs)       = delta_T * Eigen::Matrix3d::Identity();
        A.block<3, 3>(row_base, num_pairs + 3)   = -A_ba;
        A.block<3, 3>(row_base, num_pairs + 6)   = g_coeff_scalar * Eigen::Matrix3d::Identity();
        b.segment<3>(row_base) = b_term;

        // === DEBUGGING ===
        /*
        std::cout << "\n[Pair " << pair_idx << "]" << std::endl;
        std::cout << "s_ij dir: " << t_dir.transpose() << std::endl;
        std::cout << "v_i coeff: " << delta_T << std::endl;
        std::cout << "ba coeff:\n" << -A_ba << std::endl;
        std::cout << "g coeff (scalar): " << g_coeff_scalar << std::endl;
        std::cout << "RHS b[" << pair_idx << "]: " << b_term.transpose() << std::endl;
        */
    }

    // Resolver el sistema
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd x = svd.solve(b);
    int rank = (svd.singularValues().array() > 1e-6).count();

    // --- Verificación de condicionamiento ---
    const auto& singular_vals = svd.singularValues();


    
    std::cout << "\n=== Condicionamiento del sistema ===" << std::endl;
    //std::cout << "Valores singulares: " << singular_vals.transpose() << std::endl;
    std::cout << "Rango numérico: " << rank << " / " << A.cols() << std::endl;
    std::cout << "¿Sistema bien condicionado? "
            << (rank == A.cols() ? "Sí ✅" : "No ⚠️") << std::endl;

    

    std::vector<double> scales;
    for (int i = 0; i < num_pairs; ++i)
        scales.push_back(x(i));

    Eigen::Vector3d v0 = x.segment<3>(num_pairs);
    Eigen::Vector3d ba = x.segment<3>(num_pairs + 3);
    Eigen::Vector3d g  = x.segment<3>(num_pairs + 6);

    return {v0, ba, g, scales, rank, static_cast<int>(A.cols())};


}
