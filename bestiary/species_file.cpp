#include "species_file.h"

#include <toml++/toml.hpp>
#include <fstream>
#include <cstdio>
#include <cstring>

namespace bestiary {

static toml::table parse_toml(const std::filesystem::path& path)
{
    try {
        return toml::parse_file(path.string());
    } catch (const toml::parse_error& err) {
        std::fprintf(stderr, "TOML parse error in %s: %.*s\n",
                     path.string().c_str(),
                     static_cast<int>(err.description().size()),
                     err.description().data());
        return {};
    }
}

std::string detect_species_kind(const std::filesystem::path& path)
{
    auto tbl = parse_toml(path);
    if (auto v = tbl["species"]["kind"].value<std::string>())
        return *v;
    return "";
}

static std::string fmt(float v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6f", static_cast<double>(v));
    char* end = buf + std::strlen(buf) - 1;
    while (*end == '0' && *(end - 1) != '.') --end;
    *(end + 1) = '\0';
    return buf;
}

bool save_clump(const std::filesystem::path& path,
                const ClumpParams& params,
                const std::string& name,
                const ClumpExpression* expr)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.string().c_str());
        return false;
    }

    out << "[species]\n"
        << "kind = 'grass_clump'\n"
        << "name = '" << name << "'\n"
        << "\n"
        << "[morphology]\n"
        << "blade_count = "  << params.blade_count << "\n"
        << "blade_height = " << fmt(params.blade_height) << "\n"
        << "blade_width = "  << fmt(params.blade_width)  << "\n"
        << "splay_angle = "  << fmt(params.splay_angle)  << "\n"
        << "clump_radius = " << fmt(params.clump_radius) << "\n"
        << "\n"
        << "[appearance]\n"
        << "base_color = [" << fmt(params.base_color[0]) << ", "
                            << fmt(params.base_color[1]) << ", "
                            << fmt(params.base_color[2]) << "]\n";

    if (expr) {
        auto write_range = [&](const char* field, const ParamRange& r) {
            if (r.enabled) {
                out << "\n[expression." << field << "]\n"
                    << "low = "  << fmt(r.low)  << "\n"
                    << "high = " << fmt(r.high) << "\n";
            }
        };
        write_range("blade_count",  expr->blade_count);
        write_range("blade_height", expr->blade_height);
        write_range("blade_width",  expr->blade_width);
        write_range("splay_angle",  expr->splay_angle);
        write_range("clump_radius", expr->clump_radius);

        if (expr->vary_color) {
            out << "\n[expression.color]\n"
                << "dry = [" << fmt(expr->dry_color[0]) << ", "
                             << fmt(expr->dry_color[1]) << ", "
                             << fmt(expr->dry_color[2]) << "]\n"
                << "wet = [" << fmt(expr->wet_color[0]) << ", "
                             << fmt(expr->wet_color[1]) << ", "
                             << fmt(expr->wet_color[2]) << "]\n";
        }
    }

    return out.good();
}

bool load_clump(const std::filesystem::path& path,
                ClumpParams& params,
                std::string& name,
                ClumpExpression* expr)
{
    auto tbl = parse_toml(path);
    if (tbl.empty()) return false;

    if (auto sp = tbl["species"]["name"].value<std::string>())
        name = *sp;

    if (auto m = tbl["morphology"].as_table()) {
        if (auto v = (*m)["blade_count"].value<int>())      params.blade_count  = *v;
        if (auto v = (*m)["blade_height"].value<double>())   params.blade_height = static_cast<float>(*v);
        if (auto v = (*m)["blade_width"].value<double>())    params.blade_width  = static_cast<float>(*v);
        if (auto v = (*m)["splay_angle"].value<double>())    params.splay_angle  = static_cast<float>(*v);
        if (auto v = (*m)["clump_radius"].value<double>())   params.clump_radius = static_cast<float>(*v);
    }

    if (auto arr = tbl["appearance"]["base_color"].as_array(); arr && arr->size() >= 3) {
        for (int i = 0; i < 3; ++i) {
            if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                params.base_color[i] = static_cast<float>(*v);
        }
    }

    if (expr) {
        *expr = ClumpExpression{};

        auto read_range = [&](const char* field, ParamRange& r) {
            if (auto t = tbl["expression"][field].as_table()) {
                r.enabled = true;
                if (auto v = (*t)["low"].value<double>())  r.low  = static_cast<float>(*v);
                if (auto v = (*t)["high"].value<double>()) r.high = static_cast<float>(*v);
            }
        };
        read_range("blade_count",  expr->blade_count);
        read_range("blade_height", expr->blade_height);
        read_range("blade_width",  expr->blade_width);
        read_range("splay_angle",  expr->splay_angle);
        read_range("clump_radius", expr->clump_radius);

        if (auto t = tbl["expression"]["color"].as_table()) {
            expr->vary_color = true;
            if (auto arr = (*t)["dry"].as_array(); arr && arr->size() >= 3) {
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->dry_color[i] = static_cast<float>(*v);
            }
            if (auto arr = (*t)["wet"].as_array(); arr && arr->size() >= 3) {
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->wet_color[i] = static_cast<float>(*v);
            }
        }
    }

    return true;
}

bool save_bush(const std::filesystem::path& path,
               const BushParams& params,
               const std::string& name,
               const BushExpression* expr)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.string().c_str());
        return false;
    }

    out << "[species]\n"
        << "kind = 'bush'\n"
        << "name = '" << name << "'\n"
        << "\n"
        << "[morphology]\n"
        << "leaf_count = "  << params.leaf_count << "\n"
        << "leaf_length = " << fmt(params.leaf_length) << "\n"
        << "leaf_width = "  << fmt(params.leaf_width)  << "\n"
        << "bush_height = " << fmt(params.bush_height) << "\n"
        << "bush_radius = " << fmt(params.bush_radius) << "\n"
        << "stem_height = " << fmt(params.stem_height) << "\n"
        << "n_stems = "         << params.n_stems << "\n"
        << "attractor_count = " << params.attractor_count << "\n"
        << "kill_ratio = "      << fmt(params.kill_ratio) << "\n"
        << "influence_ratio = " << fmt(params.influence_ratio) << "\n"
        << "tropism = "         << fmt(params.tropism) << "\n"
        << "surface_bias = "    << fmt(params.surface_bias) << "\n"
        << "envelope_shape = "  << params.envelope_shape << "\n"
        << "branch_taper = "      << fmt(params.branch_taper) << "\n"
        << "branch_width_min = " << fmt(params.branch_width_min) << "\n"
        << "branch_width_max = " << fmt(params.branch_width_max) << "\n"
        << "branch_gravity = "   << fmt(params.branch_gravity) << "\n"
        << "branch_wobble = "    << fmt(params.branch_wobble) << "\n"
        << "tip_leaf_bias = "    << fmt(params.tip_leaf_bias) << "\n"
        << "leaf_droop = "       << fmt(params.leaf_droop) << "\n"
        << "\n"
        << "[appearance]\n"
        << "base_color = [" << fmt(params.base_color[0]) << ", "
                            << fmt(params.base_color[1]) << ", "
                            << fmt(params.base_color[2]) << "]\n";

    if (expr) {
        auto write_range = [&](const char* field, const ParamRange& r) {
            if (r.enabled) {
                out << "\n[expression." << field << "]\n"
                    << "low = "  << fmt(r.low)  << "\n"
                    << "high = " << fmt(r.high) << "\n";
            }
        };
        write_range("leaf_count",  expr->leaf_count);
        write_range("leaf_length", expr->leaf_length);
        write_range("leaf_width",  expr->leaf_width);
        write_range("bush_height", expr->bush_height);
        write_range("bush_radius", expr->bush_radius);
        write_range("stem_height", expr->stem_height);
        write_range("n_stems",         expr->n_stems);
        write_range("attractor_count", expr->attractor_count);
        write_range("kill_ratio",      expr->kill_ratio);
        write_range("influence_ratio", expr->influence_ratio);
        write_range("tropism",         expr->tropism);
        write_range("surface_bias",    expr->surface_bias);
        write_range("branch_width_min", expr->branch_width_min);
        write_range("branch_width_max", expr->branch_width_max);
        write_range("branch_gravity",  expr->branch_gravity);
        write_range("branch_wobble",   expr->branch_wobble);
        write_range("tip_leaf_bias",   expr->tip_leaf_bias);
        write_range("leaf_droop",      expr->leaf_droop);

        if (expr->vary_color) {
            out << "\n[expression.color]\n"
                << "dry = [" << fmt(expr->dry_color[0]) << ", "
                             << fmt(expr->dry_color[1]) << ", "
                             << fmt(expr->dry_color[2]) << "]\n"
                << "wet = [" << fmt(expr->wet_color[0]) << ", "
                             << fmt(expr->wet_color[1]) << ", "
                             << fmt(expr->wet_color[2]) << "]\n";
        }
    }

    return out.good();
}

bool load_bush(const std::filesystem::path& path,
               BushParams& params,
               std::string& name,
               BushExpression* expr)
{
    auto tbl = parse_toml(path);
    if (tbl.empty()) return false;

    if (auto sp = tbl["species"]["name"].value<std::string>())
        name = *sp;

    if (auto m = tbl["morphology"].as_table()) {
        if (auto v = (*m)["leaf_count"].value<int>())         params.leaf_count  = *v;
        if (auto v = (*m)["leaf_length"].value<double>())      params.leaf_length = static_cast<float>(*v);
        if (auto v = (*m)["leaf_width"].value<double>())       params.leaf_width  = static_cast<float>(*v);
        if (auto v = (*m)["bush_height"].value<double>())      params.bush_height = static_cast<float>(*v);
        if (auto v = (*m)["bush_radius"].value<double>())      params.bush_radius = static_cast<float>(*v);
        if (auto v = (*m)["stem_height"].value<double>())      params.stem_height = static_cast<float>(*v);
        if (auto v = (*m)["n_stems"].value<int>())             params.n_stems = *v;
        if (auto v = (*m)["attractor_count"].value<int>())     params.attractor_count = *v;
        if (auto v = (*m)["kill_ratio"].value<double>())       params.kill_ratio = static_cast<float>(*v);
        if (auto v = (*m)["influence_ratio"].value<double>())  params.influence_ratio = static_cast<float>(*v);
        if (auto v = (*m)["tropism"].value<double>())          params.tropism = static_cast<float>(*v);
        if (auto v = (*m)["surface_bias"].value<double>())     params.surface_bias = static_cast<float>(*v);
        if (auto v = (*m)["envelope_shape"].value<int>())      params.envelope_shape = *v;
        if (auto v = (*m)["branch_taper"].value<double>())     params.branch_taper = static_cast<float>(*v);
        if (auto v = (*m)["branch_width_min"].value<double>()) params.branch_width_min = static_cast<float>(*v);
        if (auto v = (*m)["branch_width_max"].value<double>()) params.branch_width_max = static_cast<float>(*v);
        if (auto v = (*m)["branch_gravity"].value<double>())   params.branch_gravity = static_cast<float>(*v);
        if (auto v = (*m)["branch_wobble"].value<double>())    params.branch_wobble = static_cast<float>(*v);
        if (auto v = (*m)["tip_leaf_bias"].value<double>())    params.tip_leaf_bias = static_cast<float>(*v);
        if (auto v = (*m)["leaf_droop"].value<double>())       params.leaf_droop = static_cast<float>(*v);
    }

    if (auto arr = tbl["appearance"]["base_color"].as_array(); arr && arr->size() >= 3) {
        for (int i = 0; i < 3; ++i) {
            if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                params.base_color[i] = static_cast<float>(*v);
        }
    }

    if (expr) {
        *expr = BushExpression{};

        auto read_range = [&](const char* field, ParamRange& r) {
            if (auto t = tbl["expression"][field].as_table()) {
                r.enabled = true;
                if (auto v = (*t)["low"].value<double>())  r.low  = static_cast<float>(*v);
                if (auto v = (*t)["high"].value<double>()) r.high = static_cast<float>(*v);
            }
        };
        read_range("leaf_count",  expr->leaf_count);
        read_range("leaf_length", expr->leaf_length);
        read_range("leaf_width",  expr->leaf_width);
        read_range("bush_height", expr->bush_height);
        read_range("bush_radius", expr->bush_radius);
        read_range("stem_height", expr->stem_height);
        read_range("n_stems",         expr->n_stems);
        read_range("attractor_count", expr->attractor_count);
        read_range("kill_ratio",      expr->kill_ratio);
        read_range("influence_ratio", expr->influence_ratio);
        read_range("tropism",         expr->tropism);
        read_range("surface_bias",     expr->surface_bias);
        read_range("branch_width_min", expr->branch_width_min);
        read_range("branch_width_max", expr->branch_width_max);
        read_range("branch_gravity",   expr->branch_gravity);
        read_range("branch_wobble",    expr->branch_wobble);
        read_range("tip_leaf_bias",    expr->tip_leaf_bias);
        read_range("leaf_droop",       expr->leaf_droop);

        if (auto t = tbl["expression"]["color"].as_table()) {
            expr->vary_color = true;
            if (auto arr = (*t)["dry"].as_array(); arr && arr->size() >= 3) {
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->dry_color[i] = static_cast<float>(*v);
            }
            if (auto arr = (*t)["wet"].as_array(); arr && arr->size() >= 3) {
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->wet_color[i] = static_cast<float>(*v);
            }
        }
    }

    return true;
}

bool save_tree(const std::filesystem::path& path,
               const TreeParams& params,
               const std::string& name,
               const TreeExpression* expr)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.string().c_str());
        return false;
    }

    out << "[species]\n"
        << "kind = 'tree'\n"
        << "name = '" << name << "'\n"
        << "\n"
        << "[morphology]\n"
        << "tree_height = "     << fmt(params.tree_height) << "\n"
        << "trunk_height = "    << fmt(params.trunk_height) << "\n"
        << "crown_radius = "    << fmt(params.crown_radius) << "\n"
        << "crown_height = "    << fmt(params.crown_height) << "\n"
        << "trunk_width = "     << fmt(params.trunk_width) << "\n"
        << "leaf_count = "      << params.leaf_count << "\n"
        << "leaf_length = "     << fmt(params.leaf_length) << "\n"
        << "leaf_width = "      << fmt(params.leaf_width) << "\n"
        << "attractor_count = " << params.attractor_count << "\n"
        << "kill_ratio = "      << fmt(params.kill_ratio) << "\n"
        << "influence_ratio = " << fmt(params.influence_ratio) << "\n"
        << "tropism = "         << fmt(params.tropism) << "\n"
        << "surface_bias = "    << fmt(params.surface_bias) << "\n"
        << "crown_shape = "     << params.crown_shape << "\n"
        << "branch_taper = "      << fmt(params.branch_taper) << "\n"
        << "branch_width_min = " << fmt(params.branch_width_min) << "\n"
        << "branch_width_max = " << fmt(params.branch_width_max) << "\n"
        << "branch_gravity = "   << fmt(params.branch_gravity) << "\n"
        << "branch_wobble = "    << fmt(params.branch_wobble) << "\n"
        << "tip_leaf_bias = "    << fmt(params.tip_leaf_bias) << "\n"
        << "leaf_droop = "       << fmt(params.leaf_droop) << "\n"
        << "\n"
        << "[appearance]\n"
        << "base_color = ["  << fmt(params.base_color[0]) << ", "
                             << fmt(params.base_color[1]) << ", "
                             << fmt(params.base_color[2]) << "]\n"
        << "trunk_color = [" << fmt(params.trunk_color[0]) << ", "
                             << fmt(params.trunk_color[1]) << ", "
                             << fmt(params.trunk_color[2]) << "]\n";

    if (expr) {
        auto write_range = [&](const char* field, const ParamRange& r) {
            if (r.enabled) {
                out << "\n[expression." << field << "]\n"
                    << "low = "  << fmt(r.low)  << "\n"
                    << "high = " << fmt(r.high) << "\n";
            }
        };
        write_range("tree_height",     expr->tree_height);
        write_range("trunk_height",    expr->trunk_height);
        write_range("crown_radius",    expr->crown_radius);
        write_range("crown_height",    expr->crown_height);
        write_range("trunk_width",     expr->trunk_width);
        write_range("leaf_count",      expr->leaf_count);
        write_range("leaf_length",     expr->leaf_length);
        write_range("leaf_width",      expr->leaf_width);
        write_range("attractor_count", expr->attractor_count);
        write_range("kill_ratio",      expr->kill_ratio);
        write_range("influence_ratio", expr->influence_ratio);
        write_range("tropism",         expr->tropism);
        write_range("surface_bias",     expr->surface_bias);
        write_range("branch_taper",     expr->branch_taper);
        write_range("branch_width_min", expr->branch_width_min);
        write_range("branch_width_max", expr->branch_width_max);
        write_range("branch_gravity",   expr->branch_gravity);
        write_range("branch_wobble",    expr->branch_wobble);
        write_range("tip_leaf_bias",    expr->tip_leaf_bias);
        write_range("leaf_droop",       expr->leaf_droop);

        if (expr->vary_color) {
            out << "\n[expression.color]\n"
                << "dry = [" << fmt(expr->dry_color[0]) << ", "
                             << fmt(expr->dry_color[1]) << ", "
                             << fmt(expr->dry_color[2]) << "]\n"
                << "wet = [" << fmt(expr->wet_color[0]) << ", "
                             << fmt(expr->wet_color[1]) << ", "
                             << fmt(expr->wet_color[2]) << "]\n";
        }
    }

    return out.good();
}

bool load_tree(const std::filesystem::path& path,
               TreeParams& params,
               std::string& name,
               TreeExpression* expr)
{
    auto tbl = parse_toml(path);
    if (tbl.empty()) return false;

    if (auto sp = tbl["species"]["name"].value<std::string>())
        name = *sp;

    if (auto m = tbl["morphology"].as_table()) {
        if (auto v = (*m)["tree_height"].value<double>())      params.tree_height = static_cast<float>(*v);
        if (auto v = (*m)["trunk_height"].value<double>())     params.trunk_height = static_cast<float>(*v);
        if (auto v = (*m)["crown_radius"].value<double>())     params.crown_radius = static_cast<float>(*v);
        if (auto v = (*m)["crown_height"].value<double>())     params.crown_height = static_cast<float>(*v);
        if (auto v = (*m)["trunk_width"].value<double>())      params.trunk_width = static_cast<float>(*v);
        if (auto v = (*m)["leaf_count"].value<int>())          params.leaf_count = *v;
        if (auto v = (*m)["leaf_length"].value<double>())      params.leaf_length = static_cast<float>(*v);
        if (auto v = (*m)["leaf_width"].value<double>())       params.leaf_width = static_cast<float>(*v);
        if (auto v = (*m)["attractor_count"].value<int>())     params.attractor_count = *v;
        if (auto v = (*m)["kill_ratio"].value<double>())       params.kill_ratio = static_cast<float>(*v);
        if (auto v = (*m)["influence_ratio"].value<double>())  params.influence_ratio = static_cast<float>(*v);
        if (auto v = (*m)["tropism"].value<double>())          params.tropism = static_cast<float>(*v);
        if (auto v = (*m)["surface_bias"].value<double>())     params.surface_bias = static_cast<float>(*v);
        if (auto v = (*m)["crown_shape"].value<int>())         params.crown_shape = *v;
        if (auto v = (*m)["branch_taper"].value<double>())     params.branch_taper = static_cast<float>(*v);
        if (auto v = (*m)["branch_width_min"].value<double>()) params.branch_width_min = static_cast<float>(*v);
        if (auto v = (*m)["branch_width_max"].value<double>()) params.branch_width_max = static_cast<float>(*v);
        if (auto v = (*m)["branch_gravity"].value<double>())   params.branch_gravity = static_cast<float>(*v);
        if (auto v = (*m)["branch_wobble"].value<double>())    params.branch_wobble = static_cast<float>(*v);
        if (auto v = (*m)["tip_leaf_bias"].value<double>())    params.tip_leaf_bias = static_cast<float>(*v);
        if (auto v = (*m)["leaf_droop"].value<double>())       params.leaf_droop = static_cast<float>(*v);
    }

    auto read_color = [&](const char* key, float color[3]) {
        if (auto arr = tbl["appearance"][key].as_array(); arr && arr->size() >= 3) {
            for (int i = 0; i < 3; ++i)
                if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                    color[i] = static_cast<float>(*v);
        }
    };
    read_color("base_color",  params.base_color);
    read_color("trunk_color", params.trunk_color);

    if (expr) {
        *expr = TreeExpression{};

        auto read_range = [&](const char* field, ParamRange& r) {
            if (auto t = tbl["expression"][field].as_table()) {
                r.enabled = true;
                if (auto v = (*t)["low"].value<double>())  r.low  = static_cast<float>(*v);
                if (auto v = (*t)["high"].value<double>()) r.high = static_cast<float>(*v);
            }
        };
        read_range("tree_height",     expr->tree_height);
        read_range("trunk_height",    expr->trunk_height);
        read_range("crown_radius",    expr->crown_radius);
        read_range("crown_height",    expr->crown_height);
        read_range("trunk_width",     expr->trunk_width);
        read_range("leaf_count",      expr->leaf_count);
        read_range("leaf_length",     expr->leaf_length);
        read_range("leaf_width",      expr->leaf_width);
        read_range("attractor_count", expr->attractor_count);
        read_range("kill_ratio",      expr->kill_ratio);
        read_range("influence_ratio", expr->influence_ratio);
        read_range("tropism",         expr->tropism);
        read_range("surface_bias",     expr->surface_bias);
        read_range("branch_taper",     expr->branch_taper);
        read_range("branch_width_min", expr->branch_width_min);
        read_range("branch_width_max", expr->branch_width_max);
        read_range("branch_gravity",   expr->branch_gravity);
        read_range("branch_wobble",    expr->branch_wobble);
        read_range("tip_leaf_bias",    expr->tip_leaf_bias);
        read_range("leaf_droop",       expr->leaf_droop);

        if (auto t = tbl["expression"]["color"].as_table()) {
            expr->vary_color = true;
            if (auto arr = (*t)["dry"].as_array(); arr && arr->size() >= 3) {
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->dry_color[i] = static_cast<float>(*v);
            }
            if (auto arr = (*t)["wet"].as_array(); arr && arr->size() >= 3) {
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->wet_color[i] = static_cast<float>(*v);
            }
        }
    }

    return true;
}

bool save_lplant(const std::filesystem::path& path,
                 const LPlantParams& params,
                 const std::string& name,
                 const LPlantExpression* expr)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;

    out << "[species]\n"
        << "kind = 'lplant'\n"
        << "name = '" << name << "'\n\n"
        << "[morphology]\n"
        << "archetype = "       << static_cast<int>(params.archetype) << "\n"
        << "total_height = "    << fmt(params.total_height) << "\n"
        << "trunk_height = "    << fmt(params.trunk_height) << "\n"
        << "crown_radius = "    << fmt(params.crown_radius) << "\n"
        << "crown_height = "    << fmt(params.crown_height) << "\n"
        << "trunk_width = "     << fmt(params.trunk_width) << "\n"
        << "growth_steps = "    << params.growth_steps << "\n"
        << "branch_angle = "    << fmt(params.branch_angle) << "\n"
        << "branch_angle_var = " << fmt(params.branch_angle_var) << "\n"
        << "phyllotaxis_angle = " << fmt(params.phyllotaxis_angle) << "\n"
        << "phyllotaxis_mode = " << params.phyllotaxis_mode << "\n"
        << "whorl_count = "     << params.whorl_count << "\n"
        << "internode_length = " << fmt(params.internode_length) << "\n"
        << "length_decay = "    << fmt(params.length_decay) << "\n"
        << "lambda = "          << fmt(params.lambda) << "\n"
        << "resource_alpha = "  << fmt(params.resource_alpha) << "\n"
        << "v_threshold = "     << fmt(params.v_threshold) << "\n"
        << "attractor_count = " << params.attractor_count << "\n"
        << "kill_ratio = "      << fmt(params.kill_ratio) << "\n"
        << "influence_ratio = " << fmt(params.influence_ratio) << "\n"
        << "tropism = "         << fmt(params.tropism) << "\n"
        << "surface_bias = "    << fmt(params.surface_bias) << "\n"
        << "envelope_shape = "  << params.envelope_shape << "\n"
        << "branch_taper = "    << fmt(params.branch_taper) << "\n"
        << "branch_width_min = " << fmt(params.branch_width_min) << "\n"
        << "branch_width_max = " << fmt(params.branch_width_max) << "\n"
        << "leaf_count = "      << params.leaf_count << "\n"
        << "leaf_length = "     << fmt(params.leaf_length) << "\n"
        << "leaf_width = "      << fmt(params.leaf_width) << "\n"
        << "leaf_droop = "      << fmt(params.leaf_droop) << "\n"
        << "tip_leaf_bias = "   << fmt(params.tip_leaf_bias) << "\n"
        << "branch_gravity = "  << fmt(params.branch_gravity) << "\n"
        << "branch_wobble = "   << fmt(params.branch_wobble) << "\n\n"
        << "[appearance]\n"
        << "base_color = ["  << fmt(params.base_color[0]) << ", "
                             << fmt(params.base_color[1]) << ", "
                             << fmt(params.base_color[2]) << "]\n"
        << "trunk_color = [" << fmt(params.trunk_color[0]) << ", "
                             << fmt(params.trunk_color[1]) << ", "
                             << fmt(params.trunk_color[2]) << "]\n";

    if (expr) {
        auto write_range = [&](const char* field, const ParamRange& r) {
            if (r.enabled)
                out << "\n[expression." << field << "]\nlow = " << fmt(r.low) << "\nhigh = " << fmt(r.high) << "\n";
        };
        write_range("total_height",    expr->total_height);
        write_range("trunk_height",    expr->trunk_height);
        write_range("crown_radius",    expr->crown_radius);
        write_range("crown_height",    expr->crown_height);
        write_range("trunk_width",     expr->trunk_width);
        write_range("growth_steps",    expr->growth_steps);
        write_range("branch_angle",    expr->branch_angle);
        write_range("internode_length",expr->internode_length);
        write_range("length_decay",    expr->length_decay);
        write_range("lambda",          expr->lambda);
        write_range("attractor_count", expr->attractor_count);
        write_range("tropism",         expr->tropism);
        write_range("leaf_count",      expr->leaf_count);
        write_range("leaf_length",     expr->leaf_length);
        write_range("leaf_width",      expr->leaf_width);
        write_range("leaf_droop",      expr->leaf_droop);
        write_range("branch_gravity",  expr->branch_gravity);
        write_range("branch_wobble",   expr->branch_wobble);

        if (expr->vary_color)
            out << "\n[expression.color]\n"
                << "dry = [" << fmt(expr->dry_color[0]) << ", " << fmt(expr->dry_color[1]) << ", " << fmt(expr->dry_color[2]) << "]\n"
                << "wet = [" << fmt(expr->wet_color[0]) << ", " << fmt(expr->wet_color[1]) << ", " << fmt(expr->wet_color[2]) << "]\n";
    }
    return out.good();
}

bool load_lplant(const std::filesystem::path& path,
                 LPlantParams& params,
                 std::string& name,
                 LPlantExpression* expr)
{
    auto tbl = parse_toml(path);
    if (tbl.empty()) return false;

    if (auto sp = tbl["species"]["name"].value<std::string>()) name = *sp;

    if (auto m = tbl["morphology"].as_table()) {
        if (auto v = (*m)["archetype"].value<int>())         params.archetype = static_cast<GrowthArchetype>(*v);
        if (auto v = (*m)["total_height"].value<double>())    params.total_height = static_cast<float>(*v);
        if (auto v = (*m)["trunk_height"].value<double>())    params.trunk_height = static_cast<float>(*v);
        if (auto v = (*m)["crown_radius"].value<double>())    params.crown_radius = static_cast<float>(*v);
        if (auto v = (*m)["crown_height"].value<double>())    params.crown_height = static_cast<float>(*v);
        if (auto v = (*m)["trunk_width"].value<double>())     params.trunk_width = static_cast<float>(*v);
        if (auto v = (*m)["growth_steps"].value<int>())       params.growth_steps = *v;
        if (auto v = (*m)["branch_angle"].value<double>())    params.branch_angle = static_cast<float>(*v);
        if (auto v = (*m)["branch_angle_var"].value<double>()) params.branch_angle_var = static_cast<float>(*v);
        if (auto v = (*m)["phyllotaxis_angle"].value<double>()) params.phyllotaxis_angle = static_cast<float>(*v);
        if (auto v = (*m)["phyllotaxis_mode"].value<int>())   params.phyllotaxis_mode = *v;
        if (auto v = (*m)["whorl_count"].value<int>())        params.whorl_count = *v;
        if (auto v = (*m)["internode_length"].value<double>()) params.internode_length = static_cast<float>(*v);
        if (auto v = (*m)["length_decay"].value<double>())    params.length_decay = static_cast<float>(*v);
        if (auto v = (*m)["lambda"].value<double>())          params.lambda = static_cast<float>(*v);
        if (auto v = (*m)["resource_alpha"].value<double>())  params.resource_alpha = static_cast<float>(*v);
        if (auto v = (*m)["v_threshold"].value<double>())     params.v_threshold = static_cast<float>(*v);
        if (auto v = (*m)["attractor_count"].value<int>())    params.attractor_count = *v;
        if (auto v = (*m)["kill_ratio"].value<double>())      params.kill_ratio = static_cast<float>(*v);
        if (auto v = (*m)["influence_ratio"].value<double>()) params.influence_ratio = static_cast<float>(*v);
        if (auto v = (*m)["tropism"].value<double>())         params.tropism = static_cast<float>(*v);
        if (auto v = (*m)["surface_bias"].value<double>())    params.surface_bias = static_cast<float>(*v);
        if (auto v = (*m)["envelope_shape"].value<int>())     params.envelope_shape = *v;
        if (auto v = (*m)["branch_taper"].value<double>())    params.branch_taper = static_cast<float>(*v);
        if (auto v = (*m)["branch_width_min"].value<double>()) params.branch_width_min = static_cast<float>(*v);
        if (auto v = (*m)["branch_width_max"].value<double>()) params.branch_width_max = static_cast<float>(*v);
        if (auto v = (*m)["leaf_count"].value<int>())         params.leaf_count = *v;
        if (auto v = (*m)["leaf_length"].value<double>())     params.leaf_length = static_cast<float>(*v);
        if (auto v = (*m)["leaf_width"].value<double>())      params.leaf_width = static_cast<float>(*v);
        if (auto v = (*m)["leaf_droop"].value<double>())      params.leaf_droop = static_cast<float>(*v);
        if (auto v = (*m)["tip_leaf_bias"].value<double>())   params.tip_leaf_bias = static_cast<float>(*v);
        if (auto v = (*m)["branch_gravity"].value<double>())  params.branch_gravity = static_cast<float>(*v);
        if (auto v = (*m)["branch_wobble"].value<double>())   params.branch_wobble = static_cast<float>(*v);
    }

    auto read_color = [&](const char* key, float color[3]) {
        if (auto arr = tbl["appearance"][key].as_array(); arr && arr->size() >= 3)
            for (int i = 0; i < 3; ++i)
                if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                    color[i] = static_cast<float>(*v);
    };
    read_color("base_color", params.base_color);
    read_color("trunk_color", params.trunk_color);

    if (expr) {
        *expr = LPlantExpression{};
        auto read_range = [&](const char* field, ParamRange& r) {
            if (auto t = tbl["expression"][field].as_table()) {
                r.enabled = true;
                if (auto v = (*t)["low"].value<double>())  r.low = static_cast<float>(*v);
                if (auto v = (*t)["high"].value<double>()) r.high = static_cast<float>(*v);
            }
        };
        read_range("total_height",    expr->total_height);
        read_range("trunk_height",    expr->trunk_height);
        read_range("crown_radius",    expr->crown_radius);
        read_range("crown_height",    expr->crown_height);
        read_range("trunk_width",     expr->trunk_width);
        read_range("growth_steps",    expr->growth_steps);
        read_range("branch_angle",    expr->branch_angle);
        read_range("internode_length",expr->internode_length);
        read_range("length_decay",    expr->length_decay);
        read_range("lambda",          expr->lambda);
        read_range("attractor_count", expr->attractor_count);
        read_range("tropism",         expr->tropism);
        read_range("leaf_count",      expr->leaf_count);
        read_range("leaf_length",     expr->leaf_length);
        read_range("leaf_width",      expr->leaf_width);
        read_range("leaf_droop",      expr->leaf_droop);
        read_range("branch_gravity",  expr->branch_gravity);
        read_range("branch_wobble",   expr->branch_wobble);

        if (auto t = tbl["expression"]["color"].as_table()) {
            expr->vary_color = true;
            if (auto arr = (*t)["dry"].as_array(); arr && arr->size() >= 3)
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->dry_color[i] = static_cast<float>(*v);
            if (auto arr = (*t)["wet"].as_array(); arr && arr->size() >= 3)
                for (int i = 0; i < 3; ++i)
                    if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                        expr->wet_color[i] = static_cast<float>(*v);
        }
    }
    return true;
}

// -----------------------------------------------------------------------
// Creature (unified herbivore/predator/rabbit)
// -----------------------------------------------------------------------

static const char* archetype_string(Archetype a)
{
    switch (a) {
    case Archetype::Herbivore: return "herbivore";
    case Archetype::Predator:  return "predator";
    case Archetype::Rabbit:    return "rabbit";
    case Archetype::Bird:      return "bird";
    case Archetype::Raptor:    return "raptor";
    case Archetype::Snake:     return "snake";
    }
    return "herbivore";
}

static Archetype parse_archetype(const std::string& s)
{
    if (s == "predator") return Archetype::Predator;
    if (s == "rabbit")   return Archetype::Rabbit;
    if (s == "bird")     return Archetype::Bird;
    if (s == "raptor")   return Archetype::Raptor;
    if (s == "snake")    return Archetype::Snake;
    return Archetype::Herbivore;
}

bool save_creature(const std::filesystem::path& path,
                   const CreatureProfile& profile,
                   const std::string& name)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.string().c_str());
        return false;
    }

    out << "[species]\n"
        << "kind = '" << archetype_string(profile.archetype) << "'\n"
        << "name = '" << name << "'\n"
        << "\n"
        << "[body]\n"
        << "body_length = " << fmt(profile.body_length) << "\n"
        << "body_height = " << fmt(profile.body_height) << "\n"
        << "body_color = [" << fmt(profile.body_color[0]) << ", "
                            << fmt(profile.body_color[1]) << ", "
                            << fmt(profile.body_color[2]) << "]\n"
        << "\n"
        << "[locomotion]\n"
        << "move_speed = " << fmt(profile.move_speed) << "\n"
        << "run_speed = "  << fmt(profile.run_speed)  << "\n"
        << "turn_rate = "  << fmt(profile.turn_rate)  << "\n"
        << "\n"
        << "[energy]\n"
        << "body_mass = "        << fmt(profile.body_mass)        << "\n"
        << "basal_rate = "       << fmt(profile.basal_rate)       << "\n"
        << "locomotion_cost = "  << fmt(profile.locomotion_cost)  << "\n"
        << "hunger_threshold = " << fmt(profile.hunger_threshold) << "\n"
        << "starve_threshold = " << fmt(profile.starve_threshold) << "\n"
        << "trophic_efficiency = " << fmt(profile.trophic_efficiency) << "\n"
        << "\n"
        << "[grazing]\n"
        << "graze_radius = "      << fmt(profile.graze_radius)      << "\n"
        << "graze_consume = "     << fmt(profile.graze_consume)     << "\n"
        << "graze_min_health = " << fmt(profile.graze_min_health) << "\n"
        << "graze_duration = "    << fmt(profile.graze_duration)    << "\n"
        << "\n"
        << "[wander]\n"
        << "wander_radius = " << fmt(profile.wander_radius) << "\n"
        << "wander_jitter = " << fmt(profile.wander_jitter) << "\n"
        << "\n"
        << "[flee]\n"
        << "flee_radius = "   << fmt(profile.flee_radius)   << "\n"
        << "flee_duration = " << fmt(profile.flee_duration) << "\n";

    if (profile.archetype == Archetype::Predator) {
        out << "\n[hunting]\n"
            << "hunt_radius = "      << fmt(profile.hunt_radius)      << "\n"
            << "chase_speed = "      << fmt(profile.chase_speed)      << "\n"
            << "attack_range = "     << fmt(profile.attack_range)     << "\n"
            << "kill_energy_gain = " << fmt(profile.kill_energy_gain) << "\n"
            << "stalk_speed = "      << fmt(profile.stalk_speed)      << "\n"
            << "consume_duration = " << fmt(profile.consume_duration) << "\n";
    }

    return true;
}

bool load_creature(const std::filesystem::path& path,
                   CreatureProfile& profile,
                   std::string& name)
{
    auto tbl = parse_toml(path);
    if (tbl.empty()) return false;

    if (auto v = tbl["species"]["name"].value<std::string>())
        name = *v;

    if (auto v = tbl["species"]["kind"].value<std::string>())
        profile.archetype = parse_archetype(*v);

    auto f = [&](const char* section, const char* key, float& dst) {
        if (auto v = tbl[section][key].value<double>())
            dst = static_cast<float>(*v);
    };

    f("body", "body_length", profile.body_length);
    f("body", "body_height", profile.body_height);

    if (auto arr = tbl["body"]["body_color"].as_array(); arr && arr->size() >= 3)
        for (int i = 0; i < 3; ++i)
            if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                profile.body_color[i] = static_cast<float>(*v);

    f("locomotion", "move_speed", profile.move_speed);
    f("locomotion", "run_speed",  profile.run_speed);
    f("locomotion", "turn_rate",  profile.turn_rate);

    f("energy", "body_mass",          profile.body_mass);
    f("energy", "basal_rate",        profile.basal_rate);
    f("energy", "locomotion_cost",   profile.locomotion_cost);
    f("energy", "hunger_threshold",  profile.hunger_threshold);
    f("energy", "starve_threshold",  profile.starve_threshold);
    f("energy", "trophic_efficiency", profile.trophic_efficiency);

    f("grazing", "graze_radius",      profile.graze_radius);
    f("grazing", "graze_consume",     profile.graze_consume);
    f("grazing", "graze_min_health", profile.graze_min_health);
    f("grazing", "graze_duration",    profile.graze_duration);

    f("wander", "wander_radius", profile.wander_radius);
    f("wander", "wander_jitter", profile.wander_jitter);

    f("flee", "flee_radius",   profile.flee_radius);
    f("flee", "flee_duration", profile.flee_duration);

    if (profile.archetype == Archetype::Predator) {
        f("hunting", "hunt_radius",      profile.hunt_radius);
        f("hunting", "chase_speed",      profile.chase_speed);
        f("hunting", "attack_range",     profile.attack_range);
        f("hunting", "kill_energy_gain", profile.kill_energy_gain);
        f("hunting", "stalk_speed",      profile.stalk_speed);
        f("hunting", "consume_duration", profile.consume_duration);
    }

    return true;
}

// -----------------------------------------------------------------------
// Animal morphology (archetype-specific params)
// -----------------------------------------------------------------------

bool save_animal(const std::filesystem::path& path,
                 Archetype archetype,
                 const HerbivoreParams* herb,
                 const PredatorParams* pred,
                 const RabbitParams* rabbit,
                 const BirdParams* bird,
                 const SnakeParams* snake,
                 const std::string& name)
{
    std::ofstream out(path);
    if (!out.is_open()) return false;

    out << "[species]\n"
        << "kind = '" << archetype_string(archetype) << "'\n"
        << "name = '" << name << "'\n\n";

    switch (archetype) {
    case Archetype::Herbivore:
        if (!herb) return false;
        out << "[morphology]\n"
            << "torso_length = "     << fmt(herb->torso_length)     << "\n"
            << "torso_girth = "      << fmt(herb->torso_girth)      << "\n"
            << "neck_length = "      << fmt(herb->neck_length)      << "\n"
            << "head_length = "      << fmt(herb->head_length)      << "\n"
            << "leg_length_front = " << fmt(herb->leg_length_front) << "\n"
            << "leg_length_back = "  << fmt(herb->leg_length_back)  << "\n"
            << "leg_thickness = "    << fmt(herb->leg_thickness)    << "\n"
            << "hoof_size = "        << fmt(herb->hoof_size)        << "\n"
            << "walk_period = "      << fmt(herb->walk_period_seconds) << "\n"
            << "foot_lift = "        << fmt(herb->foot_lift_height)    << "\n\n"
            << "[appearance]\n"
            << "coat_color = [" << fmt(herb->coat_color[0]) << ", "
                                << fmt(herb->coat_color[1]) << ", "
                                << fmt(herb->coat_color[2]) << "]\n";
        break;
    case Archetype::Predator:
        if (!pred) return false;
        out << "[morphology]\n"
            << "torso_length = "     << fmt(pred->torso_length)     << "\n"
            << "chest_girth = "      << fmt(pred->chest_girth)      << "\n"
            << "waist_girth = "      << fmt(pred->waist_girth)      << "\n"
            << "neck_length = "      << fmt(pred->neck_length)      << "\n"
            << "snout_length = "     << fmt(pred->snout_length)     << "\n"
            << "head_width = "       << fmt(pred->head_width)       << "\n"
            << "leg_length_front = " << fmt(pred->leg_length_front) << "\n"
            << "leg_length_back = "  << fmt(pred->leg_length_back)  << "\n"
            << "leg_thickness = "    << fmt(pred->leg_thickness)    << "\n"
            << "paw_size = "         << fmt(pred->paw_size)         << "\n"
            << "tail_length = "      << fmt(pred->tail_length)      << "\n"
            << "walk_period = "      << fmt(pred->walk_period_seconds) << "\n"
            << "foot_lift = "        << fmt(pred->foot_lift_height)    << "\n\n"
            << "[appearance]\n"
            << "coat_color = [" << fmt(pred->coat_color[0]) << ", "
                                << fmt(pred->coat_color[1]) << ", "
                                << fmt(pred->coat_color[2]) << "]\n";
        break;
    case Archetype::Rabbit:
        if (!rabbit) return false;
        out << "[morphology]\n"
            << "body_length = "      << fmt(rabbit->body_length)      << "\n"
            << "body_girth = "       << fmt(rabbit->body_girth)       << "\n"
            << "neck_length = "      << fmt(rabbit->neck_length)      << "\n"
            << "head_length = "      << fmt(rabbit->head_length)      << "\n"
            << "head_width = "       << fmt(rabbit->head_width)       << "\n"
            << "ear_length = "       << fmt(rabbit->ear_length)       << "\n"
            << "leg_length_front = " << fmt(rabbit->leg_length_front) << "\n"
            << "leg_length_back = "  << fmt(rabbit->leg_length_back)  << "\n"
            << "leg_thickness = "    << fmt(rabbit->leg_thickness)    << "\n"
            << "paw_size = "         << fmt(rabbit->paw_size)         << "\n"
            << "tail_size = "        << fmt(rabbit->tail_size)        << "\n"
            << "hop_period = "       << fmt(rabbit->hop_period_seconds) << "\n"
            << "hop_height = "       << fmt(rabbit->hop_height)         << "\n\n"
            << "[appearance]\n"
            << "coat_color = [" << fmt(rabbit->coat_color[0]) << ", "
                                << fmt(rabbit->coat_color[1]) << ", "
                                << fmt(rabbit->coat_color[2]) << "]\n";
        break;
    case Archetype::Bird:
        if (!bird) return false;
        out << "[morphology]\n"
            << "body_length = "      << fmt(bird->body_length)      << "\n"
            << "body_girth = "       << fmt(bird->body_girth)       << "\n"
            << "neck_length = "      << fmt(bird->neck_length)      << "\n"
            << "head_size = "        << fmt(bird->head_size)        << "\n"
            << "beak_length = "      << fmt(bird->beak_length)      << "\n"
            << "wing_length = "      << fmt(bird->wing_length)      << "\n"
            << "leg_length_upper = " << fmt(bird->leg_length_upper) << "\n"
            << "leg_length_lower = " << fmt(bird->leg_length_lower) << "\n"
            << "leg_thickness = "    << fmt(bird->leg_thickness)    << "\n"
            << "foot_size = "        << fmt(bird->foot_size)        << "\n"
            << "tail_length = "      << fmt(bird->tail_length)      << "\n"
            << "walk_period = "      << fmt(bird->walk_period_seconds) << "\n"
            << "foot_lift = "        << fmt(bird->foot_lift_height)    << "\n"
            << "hop_height = "       << fmt(bird->hop_height)          << "\n\n"
            << "[appearance]\n"
            << "coat_color = [" << fmt(bird->coat_color[0]) << ", "
                                << fmt(bird->coat_color[1]) << ", "
                                << fmt(bird->coat_color[2]) << "]\n";
        break;
    case Archetype::Raptor:
        if (!bird) return false;
        out << "[morphology]\n"
            << "body_length = "      << fmt(bird->body_length)      << "\n"
            << "body_girth = "       << fmt(bird->body_girth)       << "\n"
            << "neck_length = "      << fmt(bird->neck_length)      << "\n"
            << "head_size = "        << fmt(bird->head_size)        << "\n"
            << "beak_length = "      << fmt(bird->beak_length)      << "\n"
            << "wing_length = "      << fmt(bird->wing_length)      << "\n"
            << "wing_width = "       << fmt(bird->wing_width)       << "\n"
            << "wing_taper = "       << fmt(bird->wing_taper)       << "\n"
            << "leg_length_upper = " << fmt(bird->leg_length_upper) << "\n"
            << "leg_length_lower = " << fmt(bird->leg_length_lower) << "\n"
            << "leg_thickness = "    << fmt(bird->leg_thickness)    << "\n"
            << "foot_size = "        << fmt(bird->foot_size)        << "\n"
            << "tail_length = "      << fmt(bird->tail_length)      << "\n"
            << "flap_period = "      << fmt(bird->flap_period)      << "\n"
            << "flap_amplitude = "   << fmt(bird->flap_amplitude)   << "\n"
            << "flap_sweep = "       << fmt(bird->flap_sweep)       << "\n"
            << "fly_height = "       << fmt(bird->fly_height)       << "\n\n"
            << "[appearance]\n"
            << "coat_color = [" << fmt(bird->coat_color[0]) << ", "
                                << fmt(bird->coat_color[1]) << ", "
                                << fmt(bird->coat_color[2]) << "]\n";
        break;
    case Archetype::Snake:
        if (!snake) return false;
        out << "[morphology]\n"
            << "body_length = "      << fmt(snake->body_length)      << "\n"
            << "body_thickness = "   << fmt(snake->body_thickness)   << "\n"
            << "head_width = "       << fmt(snake->head_width)       << "\n"
            << "head_length = "      << fmt(snake->head_length)      << "\n"
            << "taper_tail = "       << fmt(snake->taper_tail)       << "\n"
            << "slither_period = "   << fmt(snake->slither_period)   << "\n"
            << "slither_amplitude = " << fmt(snake->slither_amplitude) << "\n"
            << "slither_waves = "    << fmt(snake->slither_waves)    << "\n\n"
            << "[appearance]\n"
            << "coat_color = [" << fmt(snake->coat_color[0]) << ", "
                                << fmt(snake->coat_color[1]) << ", "
                                << fmt(snake->coat_color[2]) << "]\n";
        break;
    }

    return out.good();
}

bool load_animal(const std::filesystem::path& path,
                 Archetype& archetype,
                 HerbivoreParams& herb,
                 PredatorParams& pred,
                 RabbitParams& rabbit,
                 BirdParams& bird,
                 SnakeParams& snake,
                 std::string& name)
{
    auto tbl = parse_toml(path);
    if (tbl.empty()) return false;

    if (auto v = tbl["species"]["name"].value<std::string>())
        name = *v;

    std::string kind_str;
    if (auto v = tbl["species"]["kind"].value<std::string>())
        kind_str = *v;
    archetype = parse_archetype(kind_str);

    auto f = [&](const char* section, const char* key, float& dst) {
        if (auto v = tbl[section][key].value<double>())
            dst = static_cast<float>(*v);
    };

    auto read_color = [&](const char* section, const char* key, float color[3]) {
        if (auto arr = tbl[section][key].as_array(); arr && arr->size() >= 3)
            for (int i = 0; i < 3; ++i)
                if (auto v = (*arr)[static_cast<size_t>(i)].value<double>())
                    color[i] = static_cast<float>(*v);
    };

    switch (archetype) {
    case Archetype::Herbivore:
        f("morphology", "torso_length",     herb.torso_length);
        f("morphology", "torso_girth",      herb.torso_girth);
        f("morphology", "neck_length",      herb.neck_length);
        f("morphology", "head_length",      herb.head_length);
        f("morphology", "leg_length_front", herb.leg_length_front);
        f("morphology", "leg_length_back",  herb.leg_length_back);
        f("morphology", "leg_thickness",    herb.leg_thickness);
        f("morphology", "hoof_size",        herb.hoof_size);
        f("morphology", "walk_period",      herb.walk_period_seconds);
        f("morphology", "foot_lift",        herb.foot_lift_height);
        read_color("appearance", "coat_color", herb.coat_color);
        break;
    case Archetype::Predator:
        f("morphology", "torso_length",     pred.torso_length);
        f("morphology", "chest_girth",      pred.chest_girth);
        f("morphology", "waist_girth",      pred.waist_girth);
        f("morphology", "neck_length",      pred.neck_length);
        f("morphology", "snout_length",     pred.snout_length);
        f("morphology", "head_width",       pred.head_width);
        f("morphology", "leg_length_front", pred.leg_length_front);
        f("morphology", "leg_length_back",  pred.leg_length_back);
        f("morphology", "leg_thickness",    pred.leg_thickness);
        f("morphology", "paw_size",         pred.paw_size);
        f("morphology", "tail_length",      pred.tail_length);
        f("morphology", "walk_period",      pred.walk_period_seconds);
        f("morphology", "foot_lift",        pred.foot_lift_height);
        read_color("appearance", "coat_color", pred.coat_color);
        break;
    case Archetype::Rabbit:
        f("morphology", "body_length",      rabbit.body_length);
        f("morphology", "body_girth",       rabbit.body_girth);
        f("morphology", "neck_length",      rabbit.neck_length);
        f("morphology", "head_length",      rabbit.head_length);
        f("morphology", "head_width",       rabbit.head_width);
        f("morphology", "ear_length",       rabbit.ear_length);
        f("morphology", "leg_length_front", rabbit.leg_length_front);
        f("morphology", "leg_length_back",  rabbit.leg_length_back);
        f("morphology", "leg_thickness",    rabbit.leg_thickness);
        f("morphology", "paw_size",         rabbit.paw_size);
        f("morphology", "tail_size",        rabbit.tail_size);
        f("morphology", "hop_period",       rabbit.hop_period_seconds);
        f("morphology", "hop_height",       rabbit.hop_height);
        read_color("appearance", "coat_color", rabbit.coat_color);
        break;
    case Archetype::Bird:
        f("morphology", "body_length",      bird.body_length);
        f("morphology", "body_girth",       bird.body_girth);
        f("morphology", "neck_length",      bird.neck_length);
        f("morphology", "head_size",        bird.head_size);
        f("morphology", "beak_length",      bird.beak_length);
        f("morphology", "wing_length",      bird.wing_length);
        f("morphology", "leg_length_upper", bird.leg_length_upper);
        f("morphology", "leg_length_lower", bird.leg_length_lower);
        f("morphology", "leg_thickness",    bird.leg_thickness);
        f("morphology", "foot_size",        bird.foot_size);
        f("morphology", "tail_length",      bird.tail_length);
        f("morphology", "walk_period",      bird.walk_period_seconds);
        f("morphology", "foot_lift",        bird.foot_lift_height);
        f("morphology", "hop_height",       bird.hop_height);
        read_color("appearance", "coat_color", bird.coat_color);
        break;
    case Archetype::Raptor:
        f("morphology", "body_length",      bird.body_length);
        f("morphology", "body_girth",       bird.body_girth);
        f("morphology", "neck_length",      bird.neck_length);
        f("morphology", "head_size",        bird.head_size);
        f("morphology", "beak_length",      bird.beak_length);
        f("morphology", "wing_length",      bird.wing_length);
        f("morphology", "wing_width",       bird.wing_width);
        f("morphology", "wing_taper",       bird.wing_taper);
        f("morphology", "leg_length_upper", bird.leg_length_upper);
        f("morphology", "leg_length_lower", bird.leg_length_lower);
        f("morphology", "leg_thickness",    bird.leg_thickness);
        f("morphology", "foot_size",        bird.foot_size);
        f("morphology", "tail_length",      bird.tail_length);
        f("morphology", "flap_period",      bird.flap_period);
        f("morphology", "flap_amplitude",   bird.flap_amplitude);
        f("morphology", "flap_sweep",       bird.flap_sweep);
        f("morphology", "fly_height",       bird.fly_height);
        read_color("appearance", "coat_color", bird.coat_color);
        break;
    case Archetype::Snake:
        f("morphology", "body_length",      snake.body_length);
        f("morphology", "body_thickness",   snake.body_thickness);
        f("morphology", "head_width",       snake.head_width);
        f("morphology", "head_length",      snake.head_length);
        f("morphology", "taper_tail",       snake.taper_tail);
        f("morphology", "slither_period",   snake.slither_period);
        f("morphology", "slither_amplitude", snake.slither_amplitude);
        f("morphology", "slither_waves",    snake.slither_waves);
        read_color("appearance", "coat_color", snake.coat_color);
        break;
    }

    return true;
}

} // namespace bestiary
