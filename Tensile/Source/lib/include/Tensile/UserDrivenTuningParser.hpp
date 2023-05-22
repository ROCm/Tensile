#pragma once

#include <Tensile/ContractionProblem.hpp>

#include <string>
#include <vector>

namespace Tensile
{
    std::pair<ContractionProblem, int> problemFromEntries(const std::vector<std::string>& entries);
    std::vector<std::pair<ContractionProblem, int>>
        getContractionProblemsFromFile(const std::string& path);
};