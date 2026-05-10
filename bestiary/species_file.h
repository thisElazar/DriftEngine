#pragma once

#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "morphology/lplant.h"
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

bool save_tree(const std::filesystem::path& path,
               const TreeParams& params,
               const std::string& name,
               const TreeExpression* expr = nullptr);

bool load_tree(const std::filesystem::path& path,
               TreeParams& params,
               std::string& name,
               TreeExpression* expr = nullptr);

bool save_lplant(const std::filesystem::path& path,
                 const LPlantParams& params,
                 const std::string& name,
                 const LPlantExpression* expr = nullptr);

bool load_lplant(const std::filesystem::path& path,
                 LPlantParams& params,
                 std::string& name,
                 LPlantExpression* expr = nullptr);

} // namespace bestiary
