#pragma once

#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "morphology/lplant.h"
#include "morphology/wildflower.h"
#include "creature/creature_profile.h"
#include "animals/herbivore.h"
#include "animals/predator.h"
#include "animals/rabbit.h"
#include "animals/bird.h"
#include "animals/snake.h"
#include <filesystem>
#include <string>

namespace bestiary {

std::string detect_species_kind(const std::filesystem::path& path);

bool save_clump(const std::filesystem::path& path,
                const ClumpParams& params,
                const std::string& name,
                const ClumpExpression* expr = nullptr,
                const char* kind_override = nullptr);

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

bool save_wildflower(const std::filesystem::path& path,
                     const WildflowerParams& params,
                     const std::string& name,
                     const WildflowerExpression* expr = nullptr);

bool load_wildflower(const std::filesystem::path& path,
                     WildflowerParams& params,
                     std::string& name,
                     WildflowerExpression* expr = nullptr);

bool save_creature(const std::filesystem::path& path,
                   const CreatureProfile& profile,
                   const std::string& name);

bool load_creature(const std::filesystem::path& path,
                   CreatureProfile& profile,
                   std::string& name);

struct NamedCreatureProfile {
    std::string      name;
    CreatureProfile  profile;
};

std::vector<NamedCreatureProfile> load_creature_dir(const std::filesystem::path& dir);

bool save_animal(const std::filesystem::path& path,
                 Archetype archetype,
                 const HerbivoreParams* herb,
                 const PredatorParams* pred,
                 const RabbitParams* rabbit,
                 const BirdParams* bird,
                 const SnakeParams* snake,
                 const std::string& name);

bool load_animal(const std::filesystem::path& path,
                 Archetype& archetype,
                 HerbivoreParams& herb,
                 PredatorParams& pred,
                 RabbitParams& rabbit,
                 BirdParams& bird,
                 SnakeParams& snake,
                 std::string& name);

} // namespace bestiary
