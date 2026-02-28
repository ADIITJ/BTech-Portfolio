#ifndef INTEGRATED_GRADIENTS_H
#define INTEGRATED_GRADIENTS_H

#include <vector>
#include "AnnexML.h"

namespace yj {
namespace xmlc {

class IntegratedGradients {
public:
    IntegratedGradients(const AnnexML* model);
    std::vector<float> computeAttributions(const std::vector<std::pair<int, float>>& input,
                                          const std::vector<std::pair<int, float>>& baseline,
                                          int target_label,
                                          int steps = 50);

private:
    const AnnexML* annex_model;
    std::vector<std::pair<int, float>> interpolateInput(const std::vector<std::pair<int, float>>& baseline,
                                                       const std::vector<std::pair<int, float>>& input,
                                                       float alpha);
};

} // namespace xmlc
} // namespace yj

#endif