// PointTrack.h
#pragma once

#include <vector>
#include <utility>
#include <Eigen/Dense>


// each point will be tracked and identified with an unique ID

struct PointTrack {
    int id;     // unique ID of the point

    // observations: pair (frame_id, point 2D), to save the view and the position.
    std::vector<std::pair<int, Eigen::Vector2d>> observations;
    
    // constructor
    PointTrack(int point_id) : id(point_id){}

    //we add an observation to the point
    void add_observation(int frame_index, const Eigen::Vector2d& pt){
        observations.emplace_back(frame_index, pt);
    }

    const std::vector<std::pair<int, Eigen::Vector2d>>& get_observations() const {
        return observations;
    }
};