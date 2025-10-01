
#include "compute_mu_from_uv.h"
#include "utils.h"
#include "bg_small_angle.h"
#include "track_manager.h"
#include "feature_tracker.h"
#include "camera_utils.h"
#include "feature_tracking.h"
#include "frame_processor.h"
#include "euroc_io.h"
#include "c2p_wrapper.h"
#include "lie_utils.h"
#include <filesystem> 
#include <pybind11/embed.h>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip> 
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <utility> 
#include <algorithm>
#include <numeric>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <sstream>


struct InitResult {
    int ref_idx; 
    int n_frames;
    Eigen::Vector3d gravity;   // gravedad
    Eigen::Vector3d velocity;    // velocity
    Eigen::Vector3d ba;        // bias acc
    double rmse_g_deg;
    double rmse_v_ms;
    double rmse_ba_ms2;
    double rmse_bg_rads;
    std::string status;          // "OK" o "Stable"
};

class CsvLogger {
public:
    explicit CsvLogger(const std::string& path, bool write_header_if_new = true)
        : path_(path)
    {
        const bool file_exists = std::filesystem::exists(path_);
        ofs_.open(path_, std::ios::out | std::ios::app);
        if (!ofs_) throw std::runtime_error("No pude abrir CSV: " + path_);
        if (write_header_if_new && !file_exists) writeHeader();
        ofs_ << std::fixed << std::setprecision(6);
    }

    void append(const InitResult& r) {
        ofs_  << r.ref_idx << ","                 
              << r.n_frames << ","
              << r.gravity.x() << "," << r.gravity.y() << "," << r.gravity.z() << ","
              << r.velocity.x()  << "," << r.velocity.y()  << "," << r.velocity.z()  << ","
              << r.ba.x()      << "," << r.ba.y()      << "," << r.ba.z()      << ","
              << r.rmse_g_deg    << ","
              << r.rmse_v_ms     << ","
              << r.rmse_ba_ms2   << ","
              << r.rmse_bg_rads  << ","
              << r.status
              << "\n";
        ofs_.flush();
    }

private:
    void writeHeader() {
        ofs_ << "ref_idx,"         
             << "n_frames,"
             << "g_w_x,g_w_y,g_w_z,"
             << "v_x,v_y,v_z,"
             << "ba_w_x,ba_w_y,ba_w_z,"
             << "rmse_g_deg,rmse_v_ms,rmse_ba_ms2,rmse_bg_rads,"
             << "status\n";
    }

    std::string path_;
    std::ofstream ofs_;
};

inline std::string infer_sequence_name(const std::string& input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    if (p.filename().empty()) {         // path con barra final
        p = p.parent_path();
    }
    return p.filename().string();
}

void preprocessAllTracks(
    const std::vector<std::string>& cam0_image_names,
    FeatureTracker& tracker,
    TrackManager& track_manager)
{
    std::cout << "[PREPROCESS] Procesando todas las imágenes..." << std::endl;
    for (int i = 0; i < cam0_image_names.size(); ++i) {
        cv::Mat image = cv::imread(cam0_image_names[i], cv::IMREAD_GRAYSCALE);
        tracker.processFrame(image, i, track_manager);
    }
    std::cout << "[PREPROCESS] Tracks completos almacenados en TrackManager." << std::endl;
}

std::vector<PointTrack> selectTracksForInit(
    TrackManager& track_manager,
    int ref_frame_idx,
    int n_frames,
    int max_tracks)
{
    // Buscar tracks que aparezcan al menos en X frames después de ref_frame_idx
    int min_obs = 4; // configurable
    auto valid_tracks = track_manager.getTracksWithMinObservations(min_obs);

    std::vector<PointTrack> subset;
    for (const auto& tr : valid_tracks) {
        bool has_ref = tr.hasObservation(ref_frame_idx);
        if (!has_ref) continue;

        bool has_future = false;
        for (int j = 1; j <= n_frames; ++j) {
            if (tr.hasObservation(ref_frame_idx + j)) {
                has_future = true;
                break;
            }
        }
        if (has_future) {
            subset.push_back(tr);
            if ((int)subset.size() >= max_tracks) break;
        }
    }
    return subset;
}


double rmse(const Eigen::Vector3d &est, const Eigen::Vector3d &gt) {
    Eigen::Vector3d diff = est - gt;
    return std::sqrt(diff.squaredNorm() / 3.0);  // 3 componentes
}

double rmse_ang(const Eigen::Vector3d &est, const Eigen::Vector3d &gt) {
    double dot = est.dot(gt);
    double norm_product = est.norm() * gt.norm();
    // Asegurarse de que el argumento de acos esté en [-1, 1]
    double cosine = std::max(-1.0, std::min(1.0, dot / norm_product));
    double angle_rad = std::acos(cosine);  // en radianes
    double angle_deg = angle_rad * 180.0 / M_PI; //en grados
    return angle_deg;
}


FeatureTracker tracker;
TrackManager track_manager;
using Clock = std::chrono::steady_clock;

namespace py = pybind11;
using namespace Eigen;

static inline double median(std::vector<double> v){
    if(v.empty()) return 0.0;

    const size_t n = v.size();
    const size_t mid = n / 2;

    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];

    if((n % 2) == 0){
        auto max_lower = std::max_element(v.begin(), v.begin() + mid);
        m = 0.5 * (m + *max_lower);
    }
    return m;

}

inline Eigen::Matrix3d quatToRot(const Eigen::Quaterniond& q_in) {
    if (!std::isfinite(q_in.w()) || !std::isfinite(q_in.x()) ||
        !std::isfinite(q_in.y()) || !std::isfinite(q_in.z())) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Quaterniond q = q_in;
    if (q.norm() == 0.0) return Eigen::Matrix3d::Identity();
    q.normalize(); // importante para evitar drift numérico
    return q.toRotationMatrix();
}

static inline double mean(std::vector<double> v){
    if (v.empty()) throw std::runtime_error("mean() of empty vector");
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}


bool compute_translation_parallax_score(
        const cv::Mat& image1,
        const cv::Mat& image2,
        const Matrix3d& K,
        const Matrix3d& R_k_k1,
        double& parallax_median_px,
        int max_features = 800)
{

// detect corners in the first image
std::vector<cv::Point2f> pts1;
// Harris/Shi-Tomasi to obtain well-conditionated corners
cv::goodFeaturesToTrack(image1, pts1, max_features, 0.01, 12.0, cv::Mat(), 3, false, 0.04);
if (pts1.size() < 20) return false;


// track a image2 with Lucas-Kanade algorithm
std::vector<cv::Point2f> pts2;
std::vector<unsigned char> status;
std::vector<float> err;
cv::calcOpticalFlowPyrLK(
  image1, image2, pts1, pts2, status, err,
  cv::Size(21,21), 3,
  cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01),
  0, 1e-4); // it gives me estimated position of pts1[i] in image2, status TRUE if succesfull

std::vector<double> residuals;
residuals.reserve(pts1.size());

for (size_t i_c = 0; i_c < pts1.size(); ++i_c){
    if(!status[i_c]) continue;
    Vector3d x1_h(pts1[i_c].x, pts1[i_c].y, 1.0);   //hom coordinates in image1
    Vector3d x2_h = K * R_k_k1.transpose() * K.inverse() * x1_h; // ray1=Kinv*x1 , then y rotate it. I project this ray with K, so i have the prediction in image2 without translation.
    if(std::abs(x2_h.z()) < 1e-12) continue;

    Vector2d x2_hat(x2_h.x() / x2_h.z() , x2_h.y() / x2_h.z()); // from hom to normal. (u2​,v2​)=(x/w,y/w)
    Vector2d x2_obs(pts2[i_c].x , pts2[i_c].y); //Lucas Kanade observation

    double r = (x2_obs - x2_hat).norm(); //[px]
    residuals.push_back(r);
}

if(residuals.size() < 20) return false;

parallax_median_px = median(residuals); //returning median to improve robustness

return true;

}

//////////////////////// Formulacion de sistema de ecuaciones //////////////////////

struct VISolution {
    Eigen::Vector3d gravity;                  
    Eigen::Vector3d velocity;                 
    Eigen::Vector3d bias;                     // sesgo estimado
    std::map<std::pair<int,int>, double> lambdas;
};

bool solveVINSWithBias(
    const std::map<int,Eigen::Matrix3d> &Gamma_j_map,
    const std::map<int,Eigen::Vector3d> &S_j_map,
    const std::map<std::pair<int,int>,Eigen::Vector3d> &mu_ij_map,
    const std::vector<double> &t,
    double g_mag,
    VISolution &solution)
{
    // determinar el conjunto de índices de características y de imágenes
    std::set<int> feature_set, image_set;
    for (const auto &kv : mu_ij_map) {
        feature_set.insert(kv.first.first);
        image_set.insert(kv.first.second);
    }
    int N = static_cast<int>(feature_set.size());
    int m = static_cast<int>(image_set.size());
    if (m < 2 || N == 0 || t.size() != static_cast<size_t>(m) || t[0] != 0.0)
        return false;

    // preparar mapeo de índices originales a índices [0..N-1] y [0..m-1]
    std::map<int,int> feat_to_idx, img_to_idx;
    int idx=0;
    for(int fi : feature_set) feat_to_idx[fi] = idx++;
    idx=0;
    for(int ji : image_set)  img_to_idx[ji] = idx++;

    // Tamaño del vector de incógnitas: G(3)+V(3)+B(3)+λ_i_j(N*m)
    int dimX = 9 + N*m;
    int numEq = 3 * (m-1) * N;
    Eigen::MatrixXd Xi = Eigen::MatrixXd::Zero(numEq, dimX);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(numEq);

    int row = 0;
    for(int j_orig = 1; j_orig < m; ++j_orig) {
        int j = j_orig; // se asume que img_to_idx mantiene el orden 0..m-1
        double tj = t[j];
        double tj2 = tj * tj;
        Eigen::Vector3d S_j = S_j_map.at(j_orig);
        Eigen::Matrix3d Gamma_j = Gamma_j_map.at(j_orig);

        // ecuación para la característica con índice lógico 0
        {
            Xi.block<3,3>(row, 0) = -0.5 * tj2 * Eigen::Matrix3d::Identity();
            Xi.block<3,3>(row, 3) = -tj * Eigen::Matrix3d::Identity();
            Xi.block<3,3>(row, 6) =  Gamma_j;
            // coeficientes de lambdas
            auto mu00 = mu_ij_map.at({*feature_set.begin(), *image_set.begin()});
            auto mu0j = mu_ij_map.at({*feature_set.begin(), j_orig});
            int idxLambda00 = 9 + feat_to_idx[*feature_set.begin()] * m + 0;
            int idxLambda0j = 9 + feat_to_idx[*feature_set.begin()] * m + j;
            Xi.block<3,1>(row, idxLambda00) =  mu00;
            Xi.block<3,1>(row, idxLambda0j) = -mu0j;
            b.segment<3>(row) = S_j;
            row += 3;
        }
        // ecuaciones para i=1..N-1
        for(auto fi_it = std::next(feature_set.begin()); fi_it != feature_set.end(); ++fi_it) {
            int i = feat_to_idx[*fi_it];
            // lambda[0,0], lambda[0,j], lambda[i,0], lambda[i,j]
            int idx00 = 9 + feat_to_idx[*feature_set.begin()] * m + 0;
            int idx0j = 9 + feat_to_idx[*feature_set.begin()] * m + j;
            int idxi0 = 9 + i * m + 0;
            int idxij = 9 + i * m + j;
            Xi.block<3,1>(row, idx00) =  mu_ij_map.at({*feature_set.begin(),*image_set.begin()});
            Xi.block<3,1>(row, idx0j) = -mu_ij_map.at({*feature_set.begin(),j_orig});
            Xi.block<3,1>(row, idxi0) = -mu_ij_map.at({*fi_it,*image_set.begin()});
            Xi.block<3,1>(row, idxij) =  mu_ij_map.at({*fi_it,j_orig});
            b.segment<3>(row) = Eigen::Vector3d::Zero();
            row += 3;
        }
    }

    // Resolver sistema Xi * X = b por mínimos cuadrados
    Eigen::VectorXd X;
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            Xi, Eigen::ComputeThinU | Eigen::ComputeThinV);
        X = svd.solve(b);
    }

    // Extraer G, V, B y normalizar gravedad
    Eigen::Vector3d G = X.segment<3>(0);
    Eigen::Vector3d V = X.segment<3>(3);
    Eigen::Vector3d B = X.segment<3>(6);
    double normG = G.norm();
    if (normG > 0) {
        double scale = g_mag / normG;
        G *= scale;
        V *= scale;
        // B no se escala al imponer |G| = g (ver el desarrollo completo si se requiere)
        for(int i=0; i<N; ++i)
            for(int j=0; j<m; ++j)
                X(9 + i*m + j) *= scale;
    }

    // Guardar resultados
    solution.gravity = G;
    solution.velocity = V;
    solution.bias = B;
    for(auto &kv : mu_ij_map) {
        int i = feat_to_idx[kv.first.first];
        int j = img_to_idx[kv.first.second];
        solution.lambdas[{kv.first.first, kv.first.second}] = X(9 + i*m + j);
    }
    return true;
}
////////////////////////////////////////////

inline std::string make_csv_basename(const std::string& seq_name, int ref_idx) {
    auto now = std::chrono::system_clock::now();
    std::time_t t  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << "results_" << seq_name
        //<< "_ref" << ref_idx << "_"
        << (tm.tm_year + 1900)
        << std::setw(2) << std::setfill('0') << tm.tm_mon + 1
        << std::setw(2) << std::setfill('0') << tm.tm_mday << "_"
        << std::setw(2) << std::setfill('0') << tm.tm_hour
        << std::setw(2) << std::setfill('0') << tm.tm_min
        << std::setw(2) << std::setfill('0') << tm.tm_sec
        << ".csv";
    return oss.str();
}

inline std::string make_results_path(const std::string& seq_name, int ref_idx) {
    namespace fs = std::filesystem;
    fs::path dir = fs::path("results");          // carpeta destino
    fs::create_directories(dir);                 // idempotente
    fs::path file = make_csv_basename(seq_name, ref_idx);
    return (dir / file).string();
}

inline std::string make_project_results_path(const std::string& seq_input_path,
                                             const std::string& seq_name,
                                             int ref_idx)
{
    namespace fs = std::filesystem;
    fs::path seq_path(seq_input_path);

    if (seq_path.filename().empty()) seq_path = seq_path.parent_path();

    fs::path data_dir   = seq_path.parent_path();      // .../data
    fs::path project_root = data_dir.parent_path();    // .../Vi-init-cpp

    fs::path outdir = project_root / "results";
    std::error_code ec;
    fs::create_directories(outdir, ec); // idempotente

    fs::path file = make_csv_basename(seq_name, ref_idx);
    return (outdir / file).string();
}




/////////////////////////////////////////////////// main ////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    py::scoped_interpreter guard{};

        if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <path_to_dataset>" << std::endl;
        return -1;
    }
    
    std::string path_to_euroc_data = argv[1];

    // Dataset containers
    std::vector<double> timu, tcam, tgt;
    std::vector<Vector3d> omega, acc, bg_gt, ba_gt, v_gt, g_gt, p_gt;
    std::vector<Quaterniond> qgt;
    std::vector<std::string> cam0_image_names;
    Matrix4d tbodycam, tbodyimu;
    int starting_frame_;


    // Load EuRoC dataset
    if (!load_euroc_sequence(path_to_euroc_data, timu, omega, acc, tcam, cam0_image_names,
                             tgt, qgt, bg_gt, ba_gt, v_gt, g_gt, p_gt, tbodycam, tbodyimu, starting_frame_)) {
        std::cerr << "Error loading EuRoC dataset." << std::endl;
        return -1;
    }
    std::cerr << "Dataset loaded." << std::endl;
    // Load camera calibration
    std::string yaml_file = path_to_euroc_data + "/mav0/cam0/sensor.yaml";
    //auto [K, distortion_params, image_size] = load_camera_calibration(yaml_file);
    auto calib = load_camera_calibration(yaml_file);
    auto& K = calib.K;
    auto& distortion_params = calib.distortion_params;
    auto& image_size = calib.image_size;
    cv::Mat K_cv; cv::eigen2cv(K, K_cv);
    //std::cout << "K matrix: " << K << std::endl;
    cv::Mat dist_cv(distortion_params);

    preprocessAllTracks(cam0_image_names, tracker, track_manager);  //preproceso todas las imagenes




    /*
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
    log_bias_file << "frame,error_small,elapsed_small_us,error_opt,elapsed_opt_us,error_constVel,elapsed_consVel_us\n";
    */

    
    // Frame-by-frame processing --> bg estimation and comparison. The result is saved in a CSV format, in results folder.
    
    /*
    for (int frame = starting_frame_; frame < 30; frame += 5) {
    //for (int frame = starting_frame_; frame < cam0_image_names.size() - 1; frame += 10) {
        process_frame_pair(frame, cam0_image_names, tcam, timu, omega, acc, 4999936e-9,
                           tgt, qgt, bg_gt, ba_gt, v_gt, g_gt, p_gt, K, K_cv, dist_cv, tbodycam, tbodyimu, log_bias_file);
    }
    */

    //    general parameters


    for(int ref_frame_idx = 1; ref_frame_idx < tcam.size(); ref_frame_idx += 250)
    {

        //int ref_frame_idx = 508;    //108
        
        std::string seq_name = infer_sequence_name(argv[1]);
        CsvLogger logger(make_project_results_path(argv[1], seq_name, ref_frame_idx));


        std::cout << "/// Processing frame " << ref_frame_idx << std::endl;
        double first_threshold = 45.0;  //45
        double first_threshold_rel = 0.7;  //0.7
        int id_threshold = 0;
        size_t N = 7;  //number of features. 11
        double ratio_new=0;
        int MAX_FRAMES = 12;
        std::map<int, double> deriv_ratio;
        std::map<int, double> times_deriv;
        deriv_ratio[1] = 100.0;
        times_deriv[1] = 0.0;
        int gt_starting_frame_ = 0;

        std::cout << "\nDetectando traslacion..." << std::endl;

        double parallax_media_px, parallax_anterior;

        for (size_t trigg_disp = 0; trigg_disp  <= 20 ; ++trigg_disp){
            id_threshold = trigg_disp;
            int actual_frame = trigg_disp + ref_frame_idx;
            const auto& img1_path = cam0_image_names[ref_frame_idx];
            const auto& img2_path = cam0_image_names[actual_frame + 1];

            cv::Mat image_1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
            cv::Mat image_2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
            if (image_1.empty() || image_2.empty()) continue;

            auto R_k_k1_opt = compute_Rij_with_c2p(
                img1_path,
                img2_path,
                K, K_cv, dist_cv);
            
            if (!R_k_k1_opt) {
                std::cout << "[C2P] without solution (few inliers) in " 
                        << actual_frame << "->" << actual_frame+1 << "\n";
                continue; // try the following pair
            }
            
            const Eigen::Matrix3d& R_k_k1 = *R_k_k1_opt;    //reference to the value
            
        auto t0 = Clock::now();
            double parallax_anterior = parallax_media_px;
            bool criteria_achieved = compute_translation_parallax_score(image_1, image_2, K, R_k_k1, parallax_media_px);
            auto t1 = Clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if(!criteria_achieved) continue;

            //std::cout << "[frame " << actual_frame << "->" << actual_frame+1
            //          << "] parallax_media_px = " << parallax_media_px << " px. Time: "<< time_ms <<" ms.\n";
            
            double parallax_rel = 0.0;
            if (parallax_anterior != 0.0) {
                parallax_rel = std::abs(parallax_media_px - parallax_anterior) / parallax_anterior;
            }
            
            std::cout << parallax_media_px << std::endl;   //average of 7ms ..... para depurar el desplazamiento y triggering
            //std::cout << time_ms << std::endl; 

            //if (parallax_media_px >= first_threshold) {
            if(((parallax_rel >= first_threshold_rel) && (trigg_disp > 2)) || ((parallax_media_px >= first_threshold) && (trigg_disp > 2))){
                std::cout << " - Criterio alcanzado en " << trigg_disp << " frames." << std::endl;
                break; // the criteria is achieved
            }

        }

        ///////////////////////////////////////////////////////////////////////////


        // ---------- estimacion de bias ------------
        auto estimate_bg_once = [&](){
            int j_bg = ref_frame_idx + 1;
            double t1_bg = tcam[ref_frame_idx];
            double t2_bg = tcam[j_bg];

            auto it1 = std::lower_bound(timu.begin(), timu.end(), t1_bg);
            auto it2 = std::upper_bound(timu.begin(), timu.end(), t2_bg);
            if (std::distance(it1, it2) < 2) {
                throw std::runtime_error("Not enough IMU data to estimate bias.");
            }

            std::vector<double> imu_times_seg(it1, it2);
            std::vector<Vector3d> omega_seg_bg, acc_seg_bg;
            {
                size_t idx1 = std::distance(timu.begin(), it1);
                size_t idx2 = std::distance(timu.begin(), it2);
                omega_seg_bg.reserve(idx2-idx1);
                acc_seg_bg.reserve(idx2-idx1);
                for (size_t k = idx1; k < idx2; ++k) {
                    omega_seg_bg.push_back(omega[k]);
                    acc_seg_bg.push_back(acc[k]);
                }
            }

            acc_seg_bg.pop_back();
            omega_seg_bg.pop_back();

            std::vector<double> dts_bg; dts_bg.reserve(imu_times_seg.size()-1);
            for (size_t i = 1; i < imu_times_seg.size(); ++i)
                dts_bg.push_back(imu_times_seg[i] - imu_times_seg[i - 1]);

            Matrix3d Rpreint_bg = Matrix3d::Identity();
            for (size_t j = 0; j < omega_seg_bg.size(); ++j)
                Rpreint_bg *= lie::ExpMap(omega_seg_bg[j] * dts_bg[j]);

            // R_ij de C2P solo i->i+1
            auto Ropt_bg = compute_Rij_with_c2p(
                cam0_image_names[ref_frame_idx],
                cam0_image_names[j_bg],
                K, K_cv, dist_cv);
            if (!Ropt_bg) throw std::runtime_error("[Bias Estimation] Could not compute R_ij i->i+1.");
            Matrix3d R01_bg = *Ropt_bg;

            double deltat_bg = std::accumulate(dts_bg.begin(), dts_bg.end(), 0.0) / double(dts_bg.size());
            return bg_small_angle(omega_seg_bg, deltat_bg, Rpreint_bg, R01_bg, tbodycam, tbodyimu);
        };

        Vector3d bg_est; 
        try { bg_est = estimate_bg_once(); }
        catch (const std::exception& e) { std::cerr << e.what() << std::endl; return -1; }
        //std::cout << "[DEBUG] bg_estimation: " << bg_est.transpose() << std::endl;

        // -----------------------------------------

        // buscamos el indice de GT que mas se acerca a la imagen0
        double min_diff = std::numeric_limits<double>::max();
        for (size_t gt_id0 = 0; gt_id0 < tcam.size(); ++gt_id0) {
            double diff = std::abs(tcam[ref_frame_idx] - tgt[gt_id0]);
            if (diff < min_diff) {
                min_diff = diff;
                gt_starting_frame_ = static_cast<int>(gt_id0);
            }
        }

        // -------------- Obtenemos valores Ground Truth -----
        double t1_gt = tgt[gt_starting_frame_];
        Eigen::Vector3d V0_gt = v_gt[gt_starting_frame_];
        Eigen::Vector3d G_gt = g_gt[gt_starting_frame_];
        Eigen::Vector3d Bg_gt = bg_gt[gt_starting_frame_];
        Eigen::Vector3d Ba_gt = ba_gt[gt_starting_frame_];
        Eigen::Matrix3d R0_gt = quatToRot(qgt[gt_starting_frame_]);
        
        std::map<int, double> historico_rmseG, historico_rmseBA, historico_rmseV;
        historico_rmseG[id_threshold-1] = 999;
        historico_rmseBA[id_threshold-1] = 999;
        historico_rmseV[id_threshold-1] = 999;
        historico_rmseG[id_threshold] = 999;
        historico_rmseBA[id_threshold] = 999;
        historico_rmseV[id_threshold] = 999;    


        for (size_t n_frames=id_threshold; n_frames <= MAX_FRAMES; ++n_frames){

            // === cada inicialización con un frame de referencia distinto ===
            int MAX_PUNTOS_CARACTERISTICOS = 20;

            auto valid_tracks = selectTracksForInit(
                track_manager, ref_frame_idx, n_frames, MAX_PUNTOS_CARACTERISTICOS);

            //std::cout << "[INIT] Usando " << valid_tracks.size() 
            //        << " tracks para inicialización desde frame " << ref_frame_idx << std::endl;


            /*
            std::cout << "Processing features in multiple images..." << std::endl;
            for (int i = 0; i < cam0_image_names.size(); ++i) {
                cv::Mat image = cv::imread(cam0_image_names[i], cv::IMREAD_GRAYSCALE);
                tracker.processFrame(image, i, track_manager);
            }

            int MAX_PUNTOS_CARACTERISTICOS = 20;
            auto valid_tracks = track_manager.getTracksWithMinObservations(4, MAX_PUNTOS_CARACTERISTICOS); //para obtener solo aquellos puntos que fueron vistos en al menos X imágenes distintas.

            std::cout << "Numero de puntos caracteristicos: " << valid_tracks.size() << std::endl;
            */

            size_t count_frames = 0;
            //size_t count_points = 0;
            
            std::map<int, Matrix3d> R_ij_map;
            std::map<int, Matrix3d> Gamma_j_map;
            std::map<int, Vector3d> S_j_map;
            std::map<std::pair<int, int>, Vector3d> mu_ij_map;
            std::map<int, double> tj_map;

            R_ij_map.clear();
            Gamma_j_map.clear();
            S_j_map.clear();
            tj_map.clear();

            // Columna de referencia (key=1)
            R_ij_map[1]    = Matrix3d::Identity();
            Gamma_j_map[1] = Matrix3d::Identity();
            S_j_map[1]     = Vector3d::Zero();
            tj_map[1]      = 0.0;
            //std::cout << "[DEBUG] 1" << std::endl;
            
            // --------- precomputo 
            std::vector<int> keys_disponibles; // guardamos qué columnas j+1 existen realmente
            for (int rel_j = 1; rel_j <= (int)n_frames; ++rel_j) {
                int abs_j = ref_frame_idx + rel_j;
                if (abs_j >= (int)tcam.size()) break;

                const double t1 = tcam[ref_frame_idx];
                const double t2 = tcam[abs_j];
                if (t2 <= t1) continue;

                // Extraer segmento IMU en [t1, t2]
                std::vector<Vector3d> omega_seg, acc_seg;
                std::vector<double> dts;
                for (size_t k = 1; k < timu.size(); ++k) {
                    if (timu[k-1] >= t1 && timu[k] <= t2) {
                        const double dt = timu[k] - timu[k-1];
                        if (dt <= 0) continue;
                        omega_seg.push_back(omega[k-1]);
                        acc_seg.push_back(acc[k-1]);
                        dts.push_back(dt);
                    }
                }
                if (dts.empty()) continue;

                Matrix3d R_ij = Matrix3d::Identity();
                Matrix3d Gamma = Matrix3d::Zero();
                Vector3d S = Vector3d::Zero();
                double tau_k = t1;

                for (size_t k = 0; k < dts.size(); ++k) {
                    const double dt = dts[k];
                    tau_k += dt;
                    const double w = t2 - tau_k;
                    if (w < 0) continue;
                    R_ij *= lie::ExpMap((omega_seg[k] - bg_est) * dt);
                    Gamma += w * R_ij * dt;
                    S     += w * R_ij * acc_seg[k] * dt;
                }

                const int key = rel_j + 1;  // columna 2..n
                R_ij_map[key]    = R_ij;
                Gamma_j_map[key] = Gamma;
                S_j_map[key]     = S;
                tj_map[key]      = (t2 - t1);
                keys_disponibles.push_back(key);
            }

            // ------ recorremos tracks (corregido) ------
            size_t count_points = 0;

            for (const auto& track : valid_tracks) {
                if (!track.hasObservation(ref_frame_idx)) continue;
                ++count_points;
                if (count_points > N) { std::cout << "All N points processed.\n"; break; }

                int i = (int)count_points; // id compacto del punto
                size_t usados = 0;

                // IMPORTANTE: obs.frame_idx es ABSOLUTO
                for (const auto& obs : track.observations) {
                    int abs_j = obs.frame_idx;
                    int rel_j = abs_j - ref_frame_idx; // relativo para nuestros mapas
                    if (rel_j < 0 || rel_j > (int)n_frames) continue;

                    // j == 0 → columna referencia
                    if (rel_j == 0) {
                        Vector3d mu_1 = computeMuFromUV(obs.uv, K);
                        if (mu_1.norm() < 1e-5) continue;
                        mu_ij_map[{i, 1}] = mu_1;
                        continue;
                    }

                    // Usar R_ij precomputado (key = rel_j+1)
                    int key = rel_j + 1;
                    auto itR = R_ij_map.find(key);
                    if (itR == R_ij_map.end()) {
                        // no tenemos integración válida para este j
                        continue;
                    }

                    // NO consultar el track con 'j' relativo; ya tenemos la obs actual
                    Vector3d mu_j = computeMuFromUV(obs.uv, K);
                    if (mu_j.norm() < 1e-5) continue;

                    Vector3d mu_ij = itR->second * mu_j;
                    mu_ij_map[{i, key}] = mu_ij;

                    // límite por frames efectivos por punto (opcional)
                    if (++usados >= n_frames) break;
                }
            }

            // -------- fin recorrido tracks (corregido) ----






            /*
            // ------------- recorremos tracks ------
            for (const auto& track : valid_tracks) {
                if (!track.hasObservation(ref_frame_idx)) continue;
                count_points++;

                if(count_points > N){
                    std::cout << "All N points processed." << std::endl;
                    break;
                }

                //std::cout << "Punto " << count_points << " en proceso." << std::endl;

                
                count_frames = 0;
                for (const auto& obs : track.observations) {
                    int j = obs.frame_idx - ref_frame_idx;      //frame donde el punto caracteristico es visto
                    int i = count_points;                       //identificacion del punto en cuestion (track.id pero normalizado)
                    Vector2d uv = obs.uv;
                    //std::cout << "uv: " << uv.transpose() << std::endl;

                    if (j < 0) continue;
                    
                    if(j==0){
                        Vector3d mu_1 = computeMuFromUV(uv, K);
                        if (mu_1.norm() < 1e-5) {
                            std::cerr << "[WARN] mu_j casi nulo para j = " << j << std::endl;
                            continue;
                        }
                        mu_ij_map[{i, 1}] = mu_1;
                        //std::cout << "mu_1: " << mu_1.transpose()<< std::endl;
                        continue;
                    }
                    

                    count_frames++;
                    if (count_frames > n_frames) break;

                // std::cout << "i=" << i << " j=" << j << std::endl;

                    // === 1. Rotación R_ij usando C2P ===
                    //auto Ropt = compute_Rij_with_c2p(
                    //    cam0_image_names[ref_frame_idx],
                    //    cam0_image_names[obs.frame_idx],
                    //    K, K_cv, dist_cv);
                    //if (!Ropt) {
                    //    std::cerr << "[C2P] Fallo al calcular R_ij para j = " << j+1 << std::endl;
                    //    continue;
                    //}
                    //Matrix3d R_ij = *Ropt;
                    //R_ij_map[{i, j+1}] = R_ij;         // no estoy seguro si necesita multiplicarse por la matrix extrinseca.
                    ////std::cout << "R_ij: " << R_ij << std::endl;
                    

                    if(count_points == 1){
                        // === 2. Intervalo temporal ===
                        double t1 = tcam[ref_frame_idx];
                        double t2 = tcam[obs.frame_idx];
                        tj_map[j+1] = t2 - t1;
                       // std::cout << "[DEBUG] 2" << std::endl;

                        // === 3. Extraer segmento IMU ===
                        std::vector<Vector3d> omega_seg, acc_seg;
                        std::vector<double> dts;
                        for (size_t ii = 1; ii < timu.size(); ++ii) {
                           // std::cout << "[DEBUG] 3" << std::endl;
                            if (timu[ii - 1] < t2 && timu[ii] > t1) {
                                double dt_imu = timu[ii] - timu[ii - 1];
                                omega_seg.push_back(omega[ii - 1]);
                                acc_seg.push_back(acc[ii - 1]);
                                dts.push_back(dt_imu);
                            }
                        }

                        if (dts.empty()) {
                            std::cerr << "[IMU] Sin muestras entre t1 = " << t1 << " y t2 = " << t2 << std::endl;
                            continue;
                        }
                        
                        //std::cout << "datos de IMU cargados exitosamente." << t1 << std::endl;

                        // === 4. Integración de Gamma y S ===
                        Matrix3d R_ij = Matrix3d::Identity();
                        Matrix3d Gamma = Matrix3d::Zero();
                        Vector3d S = Vector3d::Zero();

                        double tau_k = t1;
                        
                        for (size_t k = 0; k < dts.size(); ++k) {
                          //  std::cout << "[DEBUG] 4" << std::endl;
                            double dt = dts[k];
                            tau_k += dt;
                            double weight = t2 - tau_k;
                            if (weight < 0) continue;
                            R_ij *= lie::ExpMap((omega_seg[k] - bg_est) * dt);
                            Gamma += weight * R_ij * dt;
                            S     += weight * R_ij * acc_seg[k] * dt;
                        }
                       // std::cout << "[DEBUG] 5" << std::endl;

                        R_ij_map[j+1] = R_ij;
                        Gamma_j_map[j+1] = Gamma;
                        S_j_map[j+1] = S;
                    }

                    // === 5. Calcular mu_ij ===
                    if (!track.hasObservation(j)) {
                        std::cerr << "[ERROR] Observation j = " << j+1 << " not found in track.observations!" << std::endl;
                        continue;
                    }

                    //Vector2d uv_j = track.observations.at(j).uv;
                    Vector3d mu_j = computeMuFromUV(uv, K);

                    if (mu_j.norm() < 1e-5) {
                        std::cerr << "[WARN] mu_j casi nulo para j = " << j << std::endl;
                        continue;
                    }
                  //  std::cout << "[DEBUG] 6" << std::endl;

                    Vector3d mu_ij = R_ij_map[j+1] * mu_j;
                    mu_ij_map[{i, j+1}] = mu_ij;
                    //std::cout << "mu_ij: " << mu_ij.transpose()<< std::endl;


                    // === Log por cada par (i, j) ===
                    //std::cout << "j = " << j+1 << " — Δt: " << tj_map[j+1] << " — mu_ij.norm = " << mu_ij.norm() << std::endl;
                }
            }   
            // ----------  fin recorrido tracks
            */

            // Construir el vector de tiempos t (t[0] = 0.0; luego las diferencias):
            std::sort(keys_disponibles.begin(), keys_disponibles.end());
            keys_disponibles.erase(
                std::unique(keys_disponibles.begin(), keys_disponibles.end()),
                keys_disponibles.end()
            );

            std::vector<double> t_vals;
            t_vals.reserve(keys_disponibles.size());
            t_vals.push_back(0.0); // t0

            for (int key : keys_disponibles) {
                // key = rel_j + 1; queremos Δt_j para cada columna existente
                auto it = tj_map.find(key);
                if (it == tj_map.end()) continue;
                t_vals.push_back(it->second);
            }

            /*
            //  ------ calculamos tvals 
            std::vector<double> t_vals;
            int m_detected = 0;
            t_vals.push_back(0.0); // t0 = 0
            for (const auto &kv : tj_map) {
                // kv.first corresponde al índice j+1 y kv.second a Δt
                if (kv.first > 1) {
                    t_vals.push_back(kv.second);
                    ++m_detected;
                }
            }
            */



            // ---------- Resolver VINS con sesgo
            VISolution sol;
            double gravity_mag = 9.81; // o el valor que uses para la gravedad

            if (solveVINSWithBias(Gamma_j_map, S_j_map, mu_ij_map, t_vals, gravity_mag, sol)) {
                std::cout << "------- Results with " << n_frames << " frames -------" << std::endl;
                Eigen::Vector3d gravity = R0_gt*sol.gravity;
                Eigen::Vector3d bias_acc = R0_gt*sol.bias;
                //Eigen::Vector3d velocity_w = R0_gt*sol.velocity;  //da mayor
                //std::cout << "Gravedad [f0]: " << sol.gravity.transpose() << std::endl;
                std::cout << "Gravedad: " << gravity.transpose() << std::endl;    //world coordinates
                std::cout << "Velocidad inicial: " << sol.velocity.transpose() << std::endl;
                //std::cout << "ba estimado: " << sol.bias.transpose() << std::endl;
                std::cout << "ba estimado: " << bias_acc.transpose() << std::endl;
                // Puedes recorrer sol.lambdas para obtener las escalas λ_i_j de cada característica
                /*
                for (const auto &kv : sol.lambdas) {
                    int i = kv.first.first;
                    int j = kv.first.second;
                    double lambda_ij = kv.second;
                    std::cout << "λ(" << i << "," << j << ") = " << lambda_ij << std::endl;
                }
                */
                //-----------  Errores RMSE ---------
                double rmse_gravity = rmse_ang(gravity, G_gt);
                double rmse_velocity = rmse(sol.velocity, V0_gt);
                //double rmse_bias_a     = rmse(sol.bias, Ba_gt);
                double rmse_bias_a   = rmse(bias_acc,Ba_gt);
                double rmse_bias_g     = rmse(bg_est, Bg_gt);
                std::cout << "RMSE gravedad: " << rmse_gravity << " deg" << std::endl;
                std::cout << "RMSE velocidad: " << rmse_velocity << " m/s"   << std::endl;
                //std::cout << "RMSE bias acc: " << rmse_bias_a     << " m/s^2" << std::endl;
                std::cout << "RMSE bias acc: " << rmse_bias_a     << " m/s^2" << std::endl;
                std::cout << "RMSE bias gyr: " << rmse_bias_g     << " rad/s" << std::endl;
            
                
                
                ///// corroboramos que los errores siempre bajen
                historico_rmseG[n_frames] = rmse_gravity;
                historico_rmseBA[n_frames] = rmse_bias_a;
                historico_rmseV[n_frames] = rmse_velocity;

                //if((historico_rmseG[n_frames] - historico_rmseG[n_frames-1]) > 0.0  && (historico_rmseV[n_frames] - historico_rmseV[n_frames-1]) > 0.0 && (historico_rmseBA[n_frames] - historico_rmseBA[n_frames-1]) > 0.0 ){
                if (n_frames > id_threshold) {
                    if((historico_rmseG[n_frames] - historico_rmseG[n_frames-1]) > 0.0 ){
                        int trigg_hessian = n_frames-1;
                        std::cout << "------- Solucion estable alcanzada en (" << trigg_hessian << ") frames -----"  << std::endl;
                        std::cout << "RMSE gravedad: " << historico_rmseG[trigg_hessian] << " deg" << std::endl;
                        std::cout << "RMSE velocidad: " << historico_rmseV[trigg_hessian] << " m/s"   << std::endl;
                        std::cout << "RMSE bias acc: " << historico_rmseBA[trigg_hessian]     << " m/s^2" << std::endl;
                        std::cout << "RMSE bias gyr: " << rmse_bias_g  << " rad/s" << std::endl;
                        break;
                    }
                }
                
                // ---- CSV: guardar fila "OK"
                InitResult row_ok;
                row_ok.ref_idx      = ref_frame_idx;                     
                row_ok.n_frames     = static_cast<int>(n_frames);
                row_ok.gravity      = gravity;
                row_ok.velocity     = sol.velocity;
                row_ok.ba           = bias_acc;
                row_ok.rmse_g_deg   = rmse_gravity;
                row_ok.rmse_v_ms    = rmse_velocity;
                row_ok.rmse_ba_ms2  = rmse_bias_a;
                row_ok.rmse_bg_rads = rmse_bias_g;
                row_ok.status       = "OK";
                logger.append(row_ok);

            } else {
                std::cerr << "[Solver] No se pudo resolver el sistema visual-inercial." << std::endl;
            }
        }
    // proceso siguiente frame
    }      
    return 0;
}







////////////////////////////////////////////////////////////////////////////////////////////////////
        /*
        // -------------------------------  Building the equation system  -----------------------
        auto t0 = Clock::now();

        // Dimensiones
        int num_equations = 3 * (n_frames - 1) * (N);       // 3 filas por (i,j)
        int num_unknowns = 9 + N * n_frames;

        // Matrices del sistema
        MatrixXd Xi = MatrixXd::Zero(num_equations, num_unknowns);
        VectorXd S = VectorXd::Zero(num_equations);

        // Índice para cada lambda_j^i
        auto lambda_index = [N, n_frames](int i, int j) {
            return 9 + (i-1) * n_frames + (j-1);
        };

        // Construcción del sistema
        int row = 0;
        for (int j = 2; j <= n_frames; ++j) {
            S.segment<3>(row) = S_j_map.at(j);  //S_j_map[j]

            for (int i = 1; i <= N; ++i) {

                double tj = tj_map[j];

                // ======= Primera ecuación (vector S_j) =======
                Xi.block<3,3>(row, 0) = -(tj * tj / 2.0) * Matrix3d::Identity();    // Gravity
                Xi.block<3,3>(row, 3) = -tj * Matrix3d::Identity();                 // Velocity
                Xi.block<3,3>(row, 6) = Gamma_j_map[j];                             // Accel Bias

                Xi.block<3,1>(row, lambda_index(i, 1)) = mu_ij_map.at(std::make_pair(i, 1));
                Xi.block<3,1>(row, lambda_index(i, j)) = -mu_ij_map.at(std::make_pair(i, j));
            
                //Xi.block<3,1>(row, lambda_index(i, 1)) = mu_ij_map[{i, 1}];
                //Xi.block<3,1>(row, lambda_index(i, j)) = -mu_ij_map[{i, j}];

                // row += 3;
                // ======= Segunda ecuación (0 vector) =======
                // Xi.block<3,1>(row, lambda_index(1, 1)) = mu_ij_map[{1, 1}];
                // Xi.block<3,1>(row, lambda_index(1, j)) = -mu_ij_map[{1, j}];
                // Xi.block<3,1>(row, lambda_index(i, 1)) = -mu_ij_map[{i, 1}];
                // Xi.block<3,1>(row, lambda_index(i, j)) = mu_ij_map[{i, j}];

                row += 3;
            }
        }
        */
        // Sanidad: debe cumplirse
        //assert(row == Xi.rows());
        /*
        JacobiSVD<MatrixXd> svd(Xi, ComputeThinU | ComputeThinV);
        // Opcional: estimar el rango numérico con tolerancia
        double tol = 1e-8 * std::max(Xi.rows(), Xi.cols()) * svd.singularValues()(0);
        int rank = (svd.singularValues().array() > tol).count();
        std::cout << "Rango numérico estimado antes: " << rank << " / " << Xi.cols() << std::endl;
        

        VectorXd X = svd.solve(S);

        Vector3d g_est = X.segment<3>(0);
        Vector3d v_est = X.segment<3>(3);
        Vector3d ba_est = X.segment<3>(6);

        std::cout << "=== Solution achieved ===\n";
        std::cout << "Gravity: " << g_est.transpose() << "\n";
        std::cout << "Velocity: " << v_est.transpose() << "\n";
        std::cout << "Bias acc: " << ba_est.transpose() << "\n";

        //std::cout << "\n=== lambda_j^i ===" << std::endl;
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= n_frames; ++j) {
                int idx = 9 + (i - 1) * n_frames + (j - 1);
                double lambda = X(idx);
                //std::cout << "lambda[" << i << "][" << j << "] = " << lambda << std::endl;
            }
        }
        */

        
        /*

        // observability test

        MatrixXd H = Xi.transpose() * Xi;  // si tienes pesos, usa A = W12 * Xi; H = A.transpose()*A

        int nx = 9;
        int nl = Xi.cols() - nx;

        MatrixXd Hxx = H.topLeftCorner(nx, nx);
        MatrixXd HxL = H.topRightCorner(nx, nl);
        MatrixXd HLx = H.bottomLeftCorner(nl, nx);
        MatrixXd HLL = H.bottomRightCorner(nl, nl);

        // Resuelve HLL * X = HLx con LLT/LDLT (HLL es SPD)
        Eigen::LDLT<MatrixXd> ldlt(HLL);
        MatrixXd Sx = Hxx - HxL * ldlt.solve(HLx);

        // Espectro de la info marginal
        SelfAdjointEigenSolver<MatrixXd> esS(Sx);
        VectorXd lam = esS.eigenvalues();       // ascendente (>=0 si todo bien)
        double smin = std::sqrt(std::max(0.0, lam(0)));
        double smax = std::sqrt(std::max(0.0, lam(lam.size()-1)));
        double ratio_old = ratio_new;
        ratio_new = (smax!=0)? smin/smax : 0.0;
        deriv_ratio[n_frames] = (ratio_old!=0)? std::abs((ratio_new - ratio_old)/(ratio_old)) : 100.0;
        
        auto t1 = Clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_deriv[n_frames] = time_ms;

        std::cout << "["<< n_frames << " frames] " << "sigma_min=" << smin << " sigma_max=" << smax
                << " ratio=" << ratio_new << ". Time: "<< time_ms <<" ms.\n";


        */
        // -------------------------- End  eq system -----------------


    /*
    std::cout << "evolucion de ratio" << std::endl;
    for (const auto &[clave, valor] : deriv_ratio){
        //std::cout << clave << " -> " << valor << std::endl;
        std::cout << valor << std::endl;
    }
    std::cout << "evolucion de ratio-Times" << std::endl;
    for (const auto &[clave, valor] : times_deriv){
        //std::cout << clave << " -> " << valor << std::endl;
        std::cout << valor << std::endl;
    }
    */






// example of use:

//          ./vi_init /Users/samucerezo/dev/src/Vi-init-cpp/data/MH_01