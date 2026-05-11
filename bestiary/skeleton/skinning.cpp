#include "skeleton/skinning.h"
#include "animals/animal_mesh.h"
#include <glm/gtc/matrix_transform.hpp>

namespace bestiary {

void cpu_skin(const std::vector<AnimalVertex>& src,
              const std::vector<glm::mat4>& joint_palette,
              std::vector<SkinnedVertex>& dst)
{
    dst.resize(src.size());

    for (size_t vi = 0; vi < src.size(); ++vi) {
        const AnimalVertex& v = src[vi];
        glm::vec3 pos(0.0f);
        glm::vec3 nrm(0.0f);

        glm::vec3 rest_pos(v.position[0], v.position[1], v.position[2]);
        glm::vec3 rest_nrm(v.normal[0], v.normal[1], v.normal[2]);

        for (int b = 0; b < 4; ++b) {
            float w = static_cast<float>(v.bone_weights[b]) / 255.0f;
            if (w < 1e-6f) continue;

            const glm::mat4& m = joint_palette[v.bone_indices[b]];
            pos += w * glm::vec3(m * glm::vec4(rest_pos, 1.0f));
            nrm += w * glm::mat3(m) * rest_nrm;
        }

        nrm = glm::normalize(nrm);

        dst[vi].position[0] = pos.x;
        dst[vi].position[1] = pos.y;
        dst[vi].position[2] = pos.z;
        dst[vi].normal[0]   = nrm.x;
        dst[vi].normal[1]   = nrm.y;
        dst[vi].normal[2]   = nrm.z;
        dst[vi].color[0]    = v.color[0];
        dst[vi].color[1]    = v.color[1];
        dst[vi].color[2]    = v.color[2];
    }
}

} // namespace bestiary
