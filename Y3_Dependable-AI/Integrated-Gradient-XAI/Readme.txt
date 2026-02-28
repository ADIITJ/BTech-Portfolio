Kindly follow these steps to run, predict and explain the Integrated-Gradient
method with AnnexML on IAPRTC dataset.

Assumptions: Following is a complete step-by-step guideline for the
method and it assumes a directory called datasets with the following files :
“iaprtc-train.svm” and “iaprtc-test.svm”.

To run these commands, make sure you are in the current working directory of
Integrated Gradient Implementation.

i. Build: make -C src_integrated/annexml
​
ii. Training: src_integrated/annexml train annexml-config-ig.json
​
This saves the model with the file name “annexml-model-ig.bin” inside the
datasets directory.

iii. Testing: src_integrated/annexml predict annexml-config-ig.json
This saves the results inside the datasets directory with the file name
“annexml-result-ig.txt”.

iv. Evaluation: cat datasets/annexml-result-ig.txt | python
scripts/learning-evaluate_predictions.py

v. Explanation: src_integrated/annexml explain annexml-config-ig.json
This saves the attribution scores in the file “ig_attributions.txt” in the current
directory.

vi. Visualization of Attributions: python plot_attributions.py
​
This saves the plots inside the plots directory.