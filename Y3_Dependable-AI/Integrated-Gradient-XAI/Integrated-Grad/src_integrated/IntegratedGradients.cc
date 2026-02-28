#include "IntegratedGradients.h"
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace yj {
namespace xmlc {

IntegratedGradients::IntegratedGradients(const AnnexML* model) : annex_model(model) {}

std::vector<std::pair<int, float>> IntegratedGradients::interpolateInput(
    const std::vector<std::pair<int, float>>& baseline,
    const std::vector<std::pair<int, float>>& input,
    float alpha) {
    std::unordered_map<int, float> baseline_map, input_map;
    for (const auto& [idx, val] : baseline) baseline_map[idx] = val;
    for (const auto& [idx, val] : input) input_map[idx] = val;

    std::vector<std::pair<int, float>> interpolated;
    std::unordered_set<int> all_indices;
    for (const auto& [idx, val] : input_map) all_indices.insert(idx);
    for (const auto& [idx, val] : baseline_map) all_indices.insert(idx);

    for (int idx : all_indices) {
        float base_val = baseline_map.count(idx) ? baseline_map[idx] : 0.0;
        float in_val = input_map.count(idx) ? input_map[idx] : 0.0;
        float interp_val = base_val + alpha * (in_val - base_val);
        if (interp_val != 0.0) {
            interpolated.emplace_back(idx, interp_val);
        }
    }
    return interpolated;
}

std::vector<float> IntegratedGradients::computeAttributions(
    const std::vector<std::pair<int, float>>& input,
    const std::vector<std::pair<int, float>>& baseline,
    int target_label,
    int steps) {
    int max_fid = 0;
    for (const auto& [fid, val] : input) if (fid > max_fid) max_fid = fid;
    for (const auto& [fid, val] : baseline) if (fid > max_fid) max_fid = fid;
    std::vector<float> attributions(max_fid + 1, 0.0);

    std::vector<float> baseline_scores = annex_model->computeScores(baseline);
    float baseline_score = (static_cast<size_t>(target_label) < baseline_scores.size()) ? baseline_scores[target_label] : 0.0;

    std::vector<float> input_scores = annex_model->computeScores(input);
    float input_score = (static_cast<size_t>(target_label) < input_scores.size()) ? input_scores[target_label] : 0.0;
    float score_range = std::abs(input_score - baseline_score) + 1e-6;

    std::vector<float> sum_gradients(max_fid + 1, 0.0);
    float prev_score = baseline_score;
    for (int step = 1; step <= steps; ++step) {
        float alpha = static_cast<float>(step) / steps;
        auto interpolated = interpolateInput(baseline, input, alpha);
        std::vector<float> scores = annex_model->computeScores(interpolated);
        float score = (static_cast<size_t>(target_label) < scores.size()) ? scores[target_label] : 0.0;

        float grad = (score - prev_score) / score_range;
        for (const auto& [fid, val] : input) {
            float base_val = 0.0;
            for (const auto& [b_fid, b_val] : baseline) {
                if (b_fid == fid) { base_val = b_val; break; }
            }
            if (val != base_val) {
                float feature_contrib = val - base_val;
                sum_gradients[fid] += grad * feature_contrib;
            }
        }
        prev_score = score;
    }

    float max_abs_attr = 0.0;
    for (const auto& [fid, val] : input) {
        attributions[fid] = sum_gradients[fid];
        max_abs_attr = std::max(max_abs_attr, std::abs(attributions[fid]));
    }

    if (max_abs_attr > 0.0) {
        for (size_t fid = 0; fid < attributions.size(); ++fid) {
            attributions[fid] /= max_abs_attr;
        }
    }

    return attributions;
}

} // namespace xmlc
} // namespace yj