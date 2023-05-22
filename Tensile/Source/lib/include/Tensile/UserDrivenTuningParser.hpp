#pragma once

#include <Tensile/ContractionProblem.hpp>

#include <string>
#include <vector>

namespace Tensile
{
    std::pair<ContractionProblem, int> problemFromEntries(std::vector<std::string> entries);
    std::vector<std::pair<ContractionProblem, int>> getContractionProblemsFromFile(std::string path);
};