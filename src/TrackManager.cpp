#include "TrackManager.h"

TrackManager::TrackManager()
    : next_track_id(0) {} //we start from 0

void TrackManager::add_frame_observations(
    int frame_index,
    const std::vector<Eigen::Vector2d>& points_2d,
    const std::vector<int>& matched_tracks_id)
    {
        //verify if the points are consistent
        if(points_2d.size() != matched_tracks_id.size()){
            std::cerr << "[TrackManager ERROR] Points are not consistent." << std::endl;
        }

        //we run over all observed points
        for(size_t i=0; i<points_2d.size(); i++){
            int track_id = matched_tracks_id[i];
            const Eigen::Vector2d& pt = points_2d[i];

            if(track_id >=0){
                // is the case when is a point already seen in another image, then we update existent track
                auto it = tracks.find(track_id);
                if(it != tracks.end()){
                    it ->second.add_observation(frame_index, pt);
                } else {
                    std::cerr << "[TrackManager ERROR] ID: " << track_id << "was not find in tracks. " << std::endl; 
                }
            } else{
                //is the case when a new point appear --> we create a new track
                int new_id = next_track_id++;
                PointTrack new_track(new_id);
                new_track.add_observation(frame_index, pt);
                tracks.emplace(new_id, new_track);
            }
        }
    }

std::vector<PointTrack> get_tracks_with_min_observations(int n) const{
    std::vector<PointTrack> filtered_tracks

    for (const auto& pair : tracks){
        const PointTrack& track = pair.second;

        if(track.num_observations() >= static_cast<size_t>(n)){
            filtered_tracks.push_back(track);
        }
    }

    return filtered_tracks;

}

