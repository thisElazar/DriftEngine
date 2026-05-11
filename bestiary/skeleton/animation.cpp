#include "skeleton/animation.h"
#include <glm/gtc/quaternion.hpp>

namespace bestiary {

std::vector<glm::quat> sample_walk(const WalkCycle& cycle, float phase, int joint_count)
{
    std::vector<glm::quat> result(static_cast<size_t>(joint_count),
                                   glm::quat(1.0f, 0.0f, 0.0f, 0.0f));

    int track_count = static_cast<int>(cycle.tracks.size());

    for (int j = 0; j < joint_count && j < track_count; ++j) {
        const auto& track = cycle.tracks[j];
        if (track.empty()) continue;

        if (track.size() == 1) {
            result[j] = track[0].rotation;
            continue;
        }

        int n = static_cast<int>(track.size());
        int idx_a = n - 1;
        int idx_b = 0;

        for (int k = 0; k < n; ++k) {
            if (track[k].phase > phase) {
                idx_b = k;
                idx_a = (k + n - 1) % n;
                break;
            }
            if (k == n - 1) {
                idx_a = n - 1;
                idx_b = 0;
            }
        }

        float pa = track[idx_a].phase;
        float pb = track[idx_b].phase;

        float gap = (pb > pa) ? (pb - pa) : (1.0f - pa + pb);
        float dist = (phase >= pa) ? (phase - pa) : (1.0f - pa + phase);
        float t = (gap > 1e-6f) ? (dist / gap) : 0.0f;

        result[j] = glm::slerp(track[idx_a].rotation, track[idx_b].rotation, t);
    }

    return result;
}

} // namespace bestiary
