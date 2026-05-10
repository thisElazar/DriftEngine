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

} // namespace bestiary
