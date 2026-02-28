//
// Copyright (C) 2017 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <cstdio>
#include <fstream> // For file output
#include "AnnexML.h"
#include "AnnexMLParameter.h"
#include "IntegratedGradients.h"
#include "DataConverter.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <train or predict or explain> <JSON file>\n", argv[0]);
    return -1;
  }

  const std::string mode(argv[1]);
  char *config_json_f = argv[2];

  std::vector<std::string> args;
  for (int i = 3; i < argc; ++i) { args.push_back(argv[i]); }

  if (mode != "train" && mode != "predict" && mode != "explain") {
    fprintf(stderr, "Mode should be 'train', 'predict', or 'explain', unknown mode: '%s'\n", mode.c_str());
    return 1;
  }
  bool load_model = (mode == "predict" || mode == "explain") ? true : false;

  yj::xmlc::AnnexMLParameter param;
  if (param.ReadFromJSONFile(config_json_f) < 0) {
    fprintf(stderr, "Fail to read JSON config file\n");
    return 1;
  }
  if (param.UpdateFromArgs(args) < 0) {
    fprintf(stderr, "Fail to read args\n");
    return 1;
  }

  yj::xmlc::AnnexML classifier;
  int ret = classifier.Init(param, load_model);
  if (ret <= 0) { fprintf(stderr, "Init fail!\n"); return 1; }

  if (mode == "train") {
    classifier.Train();
  } else if (mode == "predict") {
    classifier.Predict();
  } else if (mode == "explain") {
    // Load test data
    std::vector<std::vector<std::pair<int, float>>> data_vec;
    std::vector<std::vector<int>> label_vec;
    int max_fid, max_lid;
    ret = yj::xmlc::DataConverter::ParseForFile(param.predict_file(), &data_vec, &label_vec, &max_fid, &max_lid);
    if (ret != 1) {
      fprintf(stderr, "Fail to load predict_file: '%s'\n", param.predict_file().c_str());
      return -1;
    }

    // Use first test sample for explanation
    auto input = data_vec[0];
    std::vector<std::pair<int, float>> baseline(input.size(), {0, 0.0}); // Zero baseline
    int target_label = label_vec[0][0]; // Use first true label as target

    yj::xmlc::IntegratedGradients ig(&classifier);
    std::vector<float> attributions = ig.computeAttributions(input, baseline, target_label, 50);

    // Save attributions to a file
    std::ofstream out_file("ig_attributions.txt");
    if (!out_file.is_open()) {
      fprintf(stderr, "Failed to open ig_attributions.txt for writing\n");
      return -1;
    }
    for (size_t i = 0; i < attributions.size(); ++i) {
      if (attributions[i] != 0.0) {  // Only write non-zero attributions
        out_file << i << " " << attributions[i] << "\n";
      }
    }
    out_file.close();
    fprintf(stderr, "Integrated Gradients attributions saved to ig_attributions.txt\n");
  }

  return 0;
}