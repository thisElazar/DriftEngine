#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace bestiary {

template <typename T>
struct SpatialGrid {
    float origin_x, origin_z;
    float cell_size;
    int   cols, rows;
    std::vector<std::vector<uint32_t>> cells;

    void init(float ox, float oz, float total_w, float total_h, float cs) {
        origin_x  = ox;
        origin_z  = oz;
        cell_size = cs;
        cols = std::max(1, static_cast<int>(std::ceil(total_w / cs)));
        rows = std::max(1, static_cast<int>(std::ceil(total_h / cs)));
        cells.resize(static_cast<size_t>(cols * rows));
    }

    void clear() {
        for (auto& c : cells) c.clear();
    }

    int cell_index(float x, float z) const {
        int cx = static_cast<int>((x - origin_x) / cell_size);
        int cz = static_cast<int>((z - origin_z) / cell_size);
        cx = std::clamp(cx, 0, cols - 1);
        cz = std::clamp(cz, 0, rows - 1);
        return cz * cols + cx;
    }

    void insert(uint32_t id, float x, float z) {
        cells[static_cast<size_t>(cell_index(x, z))].push_back(id);
    }

    template <typename Fn>
    void query_radius(float x, float z, float radius, Fn&& fn) const {
        int min_cx = std::clamp(static_cast<int>((x - radius - origin_x) / cell_size), 0, cols - 1);
        int max_cx = std::clamp(static_cast<int>((x + radius - origin_x) / cell_size), 0, cols - 1);
        int min_cz = std::clamp(static_cast<int>((z - radius - origin_z) / cell_size), 0, rows - 1);
        int max_cz = std::clamp(static_cast<int>((z + radius - origin_z) / cell_size), 0, rows - 1);

        for (int cz = min_cz; cz <= max_cz; ++cz) {
            for (int cx = min_cx; cx <= max_cx; ++cx) {
                const auto& cell = cells[static_cast<size_t>(cz * cols + cx)];
                for (uint32_t id : cell) {
                    fn(id);
                }
            }
        }
    }
};

} // namespace bestiary
