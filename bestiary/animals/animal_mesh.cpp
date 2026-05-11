#include "animals/animal_mesh.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <cstring>

namespace bestiary {

static constexpr float PI = 3.14159265f;

void add_joint(Skeleton& skel, int parent, const glm::vec3& offset, const char* name)
{
    Joint j;
    j.parent     = parent;
    j.local_bind = glm::translate(glm::mat4(1.0f), offset);
    std::strncpy(j.name, name, sizeof(j.name) - 1);
    j.name[sizeof(j.name) - 1] = '\0';
    skel.joints.push_back(j);
}

std::vector<glm::mat4> compute_world(const Skeleton& skel)
{
    std::vector<glm::mat4> w(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i) {
        w[i] = (skel.joints[i].parent == -1)
            ? skel.joints[i].local_bind
            : w[skel.joints[i].parent] * skel.joints[i].local_bind;
    }
    return w;
}

glm::vec3 jpos(const glm::mat4& m) { return glm::vec3(m[3]); }

void make_frame(const glm::vec3& axis, glm::vec3& right, glm::vec3& up)
{
    glm::vec3 ref = (std::abs(axis.y) < 0.99f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    right = glm::normalize(glm::cross(ref, axis));
    up    = glm::cross(axis, right);
}

void append_tube(AnimalMesh& mesh,
                 const glm::vec3& start, const glm::vec3& end,
                 float r_start, float r_end,
                 const float color[3], uint8_t bone,
                 int slices, int segments)
{
    glm::vec3 dir = end - start;
    float len = glm::length(dir);
    if (len < 1e-6f) return;

    glm::vec3 axis = dir / len;
    glm::vec3 right, up;
    make_frame(axis, right, up);

    auto base = static_cast<uint32_t>(mesh.vertices.size());

    for (int i = 0; i <= segments; ++i) {
        float t  = static_cast<float>(i) / static_cast<float>(segments);
        glm::vec3 center = glm::mix(start, end, t);
        float radius = glm::mix(r_start, r_end, t);

        for (int j = 0; j < slices; ++j) {
            float angle = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
            glm::vec3 n = right * std::cos(angle) + up * std::sin(angle);
            glm::vec3 pos = center + n * radius;

            AnimalVertex v{};
            v.position[0] = pos.x; v.position[1] = pos.y; v.position[2] = pos.z;
            v.normal[0]   = n.x;   v.normal[1]   = n.y;   v.normal[2]   = n.z;
            v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2];
            v.bone_indices[0] = bone;
            v.bone_weights[0] = 255;
            mesh.vertices.push_back(v);
        }
    }

    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = base + static_cast<uint32_t>(i * slices + j);
            uint32_t b = base + static_cast<uint32_t>(i * slices + (j + 1) % slices);
            uint32_t c = base + static_cast<uint32_t>((i + 1) * slices + j);
            uint32_t d = base + static_cast<uint32_t>((i + 1) * slices + (j + 1) % slices);
            mesh.indices.push_back(a); mesh.indices.push_back(b); mesh.indices.push_back(c);
            mesh.indices.push_back(b); mesh.indices.push_back(d); mesh.indices.push_back(c);
        }
    }
}

void append_tube_ellipse(AnimalMesh& mesh,
                         const glm::vec3& start, const glm::vec3& end,
                         float rx_start, float ry_start,
                         float rx_end, float ry_end,
                         const float color[3], uint8_t bone,
                         int slices, int segments)
{
    glm::vec3 dir = end - start;
    float len = glm::length(dir);
    if (len < 1e-6f) return;

    glm::vec3 axis = dir / len;
    glm::vec3 right, up;
    make_frame(axis, right, up);

    auto base = static_cast<uint32_t>(mesh.vertices.size());

    for (int i = 0; i <= segments; ++i) {
        float t  = static_cast<float>(i) / static_cast<float>(segments);
        glm::vec3 center = glm::mix(start, end, t);
        float rx = glm::mix(rx_start, rx_end, t);
        float ry = glm::mix(ry_start, ry_end, t);

        for (int j = 0; j < slices; ++j) {
            float angle = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
            float ca = std::cos(angle), sa = std::sin(angle);
            glm::vec3 pos = center + right * (rx * ca) + up * (ry * sa);
            float inv_rx = (rx > 1e-6f) ? (ca / rx) : 0.0f;
            float inv_ry = (ry > 1e-6f) ? (sa / ry) : 0.0f;
            glm::vec3 n = glm::normalize(right * inv_rx + up * inv_ry);

            AnimalVertex v{};
            v.position[0] = pos.x; v.position[1] = pos.y; v.position[2] = pos.z;
            v.normal[0]   = n.x;   v.normal[1]   = n.y;   v.normal[2]   = n.z;
            v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2];
            v.bone_indices[0] = bone;
            v.bone_weights[0] = 255;
            mesh.vertices.push_back(v);
        }
    }

    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = base + static_cast<uint32_t>(i * slices + j);
            uint32_t b = base + static_cast<uint32_t>(i * slices + (j + 1) % slices);
            uint32_t c = base + static_cast<uint32_t>((i + 1) * slices + j);
            uint32_t d = base + static_cast<uint32_t>((i + 1) * slices + (j + 1) % slices);
            mesh.indices.push_back(a); mesh.indices.push_back(b); mesh.indices.push_back(c);
            mesh.indices.push_back(b); mesh.indices.push_back(d); mesh.indices.push_back(c);
        }
    }
}

void append_cap(AnimalMesh& mesh,
                const glm::vec3& center, const glm::vec3& outward,
                float radius, const float color[3], uint8_t bone_index,
                int slices, int stacks)
{
    glm::vec3 axis = glm::normalize(outward);
    glm::vec3 right, up;
    make_frame(axis, right, up);

    auto base = static_cast<uint32_t>(mesh.vertices.size());

    {
        glm::vec3 pos = center + axis * radius;
        AnimalVertex v{};
        v.position[0] = pos.x; v.position[1] = pos.y; v.position[2] = pos.z;
        v.normal[0] = axis.x; v.normal[1] = axis.y; v.normal[2] = axis.z;
        v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2];
        v.bone_indices[0] = bone_index;
        v.bone_weights[0] = 255;
        mesh.vertices.push_back(v);
    }

    for (int i = 1; i <= stacks; ++i) {
        float phi = (PI * 0.5f) * static_cast<float>(i) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);

        for (int j = 0; j < slices; ++j) {
            float theta = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
            glm::vec3 n = axis * cp + (right * std::cos(theta) + up * std::sin(theta)) * sp;
            glm::vec3 pos = center + n * radius;

            AnimalVertex v{};
            v.position[0] = pos.x; v.position[1] = pos.y; v.position[2] = pos.z;
            v.normal[0] = n.x; v.normal[1] = n.y; v.normal[2] = n.z;
            v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2];
            v.bone_indices[0] = bone_index;
            v.bone_weights[0] = 255;
            mesh.vertices.push_back(v);
        }
    }

    for (int j = 0; j < slices; ++j) {
        uint32_t a = base + 1 + static_cast<uint32_t>(j);
        uint32_t b = base + 1 + static_cast<uint32_t>((j + 1) % slices);
        mesh.indices.push_back(base);
        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
    }

    for (int i = 0; i < stacks - 1; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = base + 1 + static_cast<uint32_t>(i * slices + j);
            uint32_t b = base + 1 + static_cast<uint32_t>(i * slices + (j + 1) % slices);
            uint32_t c = base + 1 + static_cast<uint32_t>((i + 1) * slices + j);
            uint32_t d = base + 1 + static_cast<uint32_t>((i + 1) * slices + (j + 1) % slices);
            mesh.indices.push_back(a); mesh.indices.push_back(c); mesh.indices.push_back(b);
            mesh.indices.push_back(b); mesh.indices.push_back(c); mesh.indices.push_back(d);
        }
    }
}

void append_sphere(AnimalMesh& mesh,
                   const glm::vec3& center, float radius,
                   const float color[3], uint8_t bone_index,
                   int slices, int stacks)
{
    auto base = static_cast<uint32_t>(mesh.vertices.size());

    auto push_vert = [&](const glm::vec3& n) {
        glm::vec3 pos = center + n * radius;
        AnimalVertex v{};
        v.position[0] = pos.x; v.position[1] = pos.y; v.position[2] = pos.z;
        v.normal[0] = n.x; v.normal[1] = n.y; v.normal[2] = n.z;
        v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2];
        v.bone_indices[0] = bone_index;
        v.bone_weights[0] = 255;
        mesh.vertices.push_back(v);
    };

    push_vert({0, 1, 0});

    for (int i = 1; i < stacks; ++i) {
        float phi = PI * static_cast<float>(i) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);
        for (int j = 0; j < slices; ++j) {
            float theta = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
            push_vert({sp * std::cos(theta), cp, sp * std::sin(theta)});
        }
    }

    push_vert({0, -1, 0});

    for (int j = 0; j < slices; ++j) {
        uint32_t a = base + 1 + static_cast<uint32_t>(j);
        uint32_t b = base + 1 + static_cast<uint32_t>((j + 1) % slices);
        mesh.indices.push_back(base);
        mesh.indices.push_back(b);
        mesh.indices.push_back(a);
    }

    for (int i = 0; i < stacks - 2; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = base + 1 + static_cast<uint32_t>(i * slices + j);
            uint32_t b = base + 1 + static_cast<uint32_t>(i * slices + (j + 1) % slices);
            uint32_t c = base + 1 + static_cast<uint32_t>((i + 1) * slices + j);
            uint32_t d = base + 1 + static_cast<uint32_t>((i + 1) * slices + (j + 1) % slices);
            mesh.indices.push_back(a); mesh.indices.push_back(b); mesh.indices.push_back(c);
            mesh.indices.push_back(b); mesh.indices.push_back(d); mesh.indices.push_back(c);
        }
    }

    uint32_t bottom = base + static_cast<uint32_t>(1 + (stacks - 1) * slices);
    uint32_t ring_base = base + 1 + static_cast<uint32_t>((stacks - 2) * slices);
    for (int j = 0; j < slices; ++j) {
        uint32_t a = ring_base + static_cast<uint32_t>(j);
        uint32_t b = ring_base + static_cast<uint32_t>((j + 1) % slices);
        mesh.indices.push_back(bottom);
        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
    }
}

void append_ellipsoid(AnimalMesh& mesh,
                      const glm::vec3& center,
                      float rx, float ry, float rz,
                      const glm::vec3& forward,
                      const float color[3], uint8_t bone_index,
                      int slices, int stacks)
{
    glm::vec3 fwd = glm::normalize(forward);
    glm::vec3 right, up;
    make_frame(fwd, right, up);

    auto base = static_cast<uint32_t>(mesh.vertices.size());

    auto push_vert = [&](float phi, float theta) {
        float sp = std::sin(phi), cp = std::cos(phi);
        float ct = std::cos(theta), st = std::sin(theta);
        glm::vec3 pos = center + right * (rx * sp * ct) + up * (ry * cp) + fwd * (rz * sp * st);
        glm::vec3 n = glm::normalize(right * (sp * ct / rx) + up * (cp / ry) + fwd * (sp * st / rz));

        AnimalVertex v{};
        v.position[0] = pos.x; v.position[1] = pos.y; v.position[2] = pos.z;
        v.normal[0] = n.x; v.normal[1] = n.y; v.normal[2] = n.z;
        v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2];
        v.bone_indices[0] = bone_index;
        v.bone_weights[0] = 255;
        mesh.vertices.push_back(v);
    };

    push_vert(0.0f, 0.0f);

    for (int i = 1; i < stacks; ++i) {
        float phi = PI * static_cast<float>(i) / static_cast<float>(stacks);
        for (int j = 0; j < slices; ++j) {
            float theta = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
            push_vert(phi, theta);
        }
    }

    push_vert(PI, 0.0f);

    for (int j = 0; j < slices; ++j) {
        uint32_t a = base + 1 + static_cast<uint32_t>(j);
        uint32_t b = base + 1 + static_cast<uint32_t>((j + 1) % slices);
        mesh.indices.push_back(base);
        mesh.indices.push_back(b);
        mesh.indices.push_back(a);
    }

    for (int i = 0; i < stacks - 2; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = base + 1 + static_cast<uint32_t>(i * slices + j);
            uint32_t b = base + 1 + static_cast<uint32_t>(i * slices + (j + 1) % slices);
            uint32_t c = base + 1 + static_cast<uint32_t>((i + 1) * slices + j);
            uint32_t d = base + 1 + static_cast<uint32_t>((i + 1) * slices + (j + 1) % slices);
            mesh.indices.push_back(a); mesh.indices.push_back(b); mesh.indices.push_back(c);
            mesh.indices.push_back(b); mesh.indices.push_back(d); mesh.indices.push_back(c);
        }
    }

    uint32_t bottom = base + static_cast<uint32_t>(1 + (stacks - 1) * slices);
    uint32_t ring_base = base + 1 + static_cast<uint32_t>((stacks - 2) * slices);
    for (int j = 0; j < slices; ++j) {
        uint32_t a = ring_base + static_cast<uint32_t>(j);
        uint32_t b = ring_base + static_cast<uint32_t>((j + 1) % slices);
        mesh.indices.push_back(bottom);
        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
    }
}

void append_path_tube(AnimalMesh& mesh,
                      const PathNode* nodes, int count,
                      const float color[3],
                      int slices, int segments_per_span)
{
    if (count < 2) return;

    auto base = static_cast<uint32_t>(mesh.vertices.size());

    for (int span = 0; span < count - 1; ++span) {
        const auto& n0 = nodes[span];
        const auto& n1 = nodes[span + 1];

        glm::vec3 dir = n1.pos - n0.pos;
        float len = glm::length(dir);
        if (len < 1e-6f) continue;
        glm::vec3 axis = dir / len;

        glm::vec3 right, up;
        make_frame(axis, right, up);

        int ring_start = (span == 0) ? 0 : 1;
        for (int i = ring_start; i <= segments_per_span; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(segments_per_span);
            glm::vec3 center = glm::mix(n0.pos, n1.pos, t);
            float radius = glm::mix(n0.radius, n1.radius, t);

            uint8_t w1 = static_cast<uint8_t>(t * 255.0f + 0.5f);
            uint8_t w0 = 255 - w1;

            for (int j = 0; j < slices; ++j) {
                float angle = 2.0f * PI * static_cast<float>(j) / static_cast<float>(slices);
                glm::vec3 n = right * std::cos(angle) + up * std::sin(angle);
                glm::vec3 pos = center + n * radius;

                AnimalVertex v{};
                v.position[0] = pos.x; v.position[1] = pos.y; v.position[2] = pos.z;
                v.normal[0] = n.x; v.normal[1] = n.y; v.normal[2] = n.z;
                v.color[0] = color[0]; v.color[1] = color[1]; v.color[2] = color[2];
                v.bone_indices[0] = n0.bone;
                v.bone_indices[1] = n1.bone;
                v.bone_weights[0] = w0;
                v.bone_weights[1] = w1;
                mesh.vertices.push_back(v);
            }
        }
    }

    int actual_rings = static_cast<int>((mesh.vertices.size() - base) / slices);
    for (int i = 0; i < actual_rings - 1; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = base + static_cast<uint32_t>(i * slices + j);
            uint32_t b = base + static_cast<uint32_t>(i * slices + (j + 1) % slices);
            uint32_t c = base + static_cast<uint32_t>((i + 1) * slices + j);
            uint32_t d = base + static_cast<uint32_t>((i + 1) * slices + (j + 1) % slices);
            mesh.indices.push_back(a); mesh.indices.push_back(b); mesh.indices.push_back(c);
            mesh.indices.push_back(b); mesh.indices.push_back(d); mesh.indices.push_back(c);
        }
    }
}

} // namespace bestiary
