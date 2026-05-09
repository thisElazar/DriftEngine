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
        if (auto v = (*m)["leaf_count"].value<int>())      params.leaf_count  = *v;
        if (auto v = (*m)["leaf_length"].value<double>())   params.leaf_length = static_cast<float>(*v);
        if (auto v = (*m)["leaf_width"].value<double>())    params.leaf_width  = static_cast<float>(*v);
        if (auto v = (*m)["bush_height"].value<double>())   params.bush_height = static_cast<float>(*v);
        if (auto v = (*m)["bush_radius"].value<double>())   params.bush_radius = static_cast<float>(*v);
        if (auto v = (*m)["stem_height"].value<double>())   params.stem_height = static_cast<float>(*v);
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

} // namespace bestiary
