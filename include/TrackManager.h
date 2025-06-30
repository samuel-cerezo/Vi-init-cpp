#pragma once

#include <unordered_map>
#include "PointTrack.h"


class TrackManager{

    public:
        TrackManager(); //constructor
        
        //declaring the method
        void add_frame_observations(
            int frame_index,
            const std::vector<Eigen::Vector2d>& points_2d,
            const std::vector<int>& matched_tracks_id);
            
        std::vector<PointTrack> get_tracks_with_min_observations(int n) const;
        
    private:
        // we save all the active tracks
        std::unordered_map<int,PointTrack> tracks;
        // for example:
        //      tracks[4] -> pointTrack with id=4

        // unique ID for new points
        int next_track_id;

};