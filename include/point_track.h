// This file define the structure that represent a point
// observed in multiple images.

#pragma once
#include <vector>
#include <Eigen/Core>

struct PointObservation {
    int frame_idx;  // índice de la imagen
    Eigen::Vector2d uv;  // coordenadas 
};

struct PointTrack {
    int id;  // ID único del punto
    std::vector<PointObservation> observations;

    void addObservation(int frame_idx, const Eigen::Vector2d& uv) {
        observations.push_back({frame_idx, uv});
    }

    size_t numObservations() const {
        return observations.size();
    }
};
