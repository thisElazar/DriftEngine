#pragma once

#include "morphology/clump.h"
#include "morphology/bush.h"
#include <filesystem>
#include <string>

namespace bestiary {

std::string detect_species_kind(const std::filesystem::path& path);

bool save_clump(const std::filesystem::path& path,
                const ClumpParams& params,
                const std::string& name,
                const ClumpExpression* expr = nullptr);

bool load_clump(const std::filesystem::path& path,
                ClumpParams& params,
                std::string& name,
                ClumpExpression* expr = nullptr);

bool save_bush(const std::filesystem::path& path,
               const BushParams& params,
               const std::string& name,
               const BushExpression* expr = nullptr);

bool load_bush(const std::filesystem::path& path,
               BushParams& params,
               std::string& name,
               BushExpression* expr = nullptr);

} // namespace bestiary
