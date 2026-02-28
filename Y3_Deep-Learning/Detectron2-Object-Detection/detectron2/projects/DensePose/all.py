import os
import subprocess

input_dir = "/Users/ashishdate/Documents/Decetron2/image"
output_dir = "/Users/ashishdate/Documents/Decetron2/output1"
config_path = "configs/densepose_rcnn_R_50_FPN_s1x.yaml"
model_path = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"dp_{filename}.txt")  # Saving logs

        print(f"Processing {filename}...")

        command = [
            "python", "apply_net.py", "show",
            config_path,
            model_path,
            input_path,
            "bbox,dp_segm",
            "-v",
            "--opts", "MODEL.DEVICE", "cpu"
        ]

        # Save output logs to file (optional)
        with open(output_path, "wb") as out_file:
            subprocess.run(command, stdout=out_file, stderr=subprocess.STDOUT)