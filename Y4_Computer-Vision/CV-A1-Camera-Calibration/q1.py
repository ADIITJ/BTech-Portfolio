import numpy as np
import cv2
import glob

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 35 

# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

obj_points = [] 
img_points = [] 
calibrated_image_paths = [] 

# Get a list of all checkerboard images
images = glob.glob('images/*.png')
if not images:
    print("Error: No PNG images found in the 'images' directory.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Could not read image {fname}. Skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    print(f"Processing {fname}: Found corners = {ret}")

    if ret:
        # Refine corners to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        obj_points.append(objp)
        img_points.append(corners)
        calibrated_image_paths.append(fname) # Store the path

        # Save the first few calibration images for report
        if len(obj_points) <= 3:
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
            cv2.imwrite(f'calibration_corners_{len(obj_points)}.png', img)
            print(f"Saved calibration_corners_{len(obj_points)}.png")
        
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        cv2.imshow('corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if not obj_points or not img_points:
    print("Error: No corners were successfully detected in any image. Calibration failed.")
    exit()

# Calibrate the camera
# The 'gray' variable from the last processed image is used for its shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("\n--- Calibration Results ---")
print("Intrinsic Matrix (K):\n", mtx)
print("\nDistortion Coefficients:\n", dist)

# Reprojection Error
mean_error = 0
for i in range(len(obj_points)):
    img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
    mean_error += error

print(f"\nAverage Reprojection Error: {mean_error / len(obj_points):.4f}")

# Calculate and display calibration quality metrics
print(f"\nCalibration Quality Assessment:")
print(f"Number of images used: {len(obj_points)}")
print(f"Image size: {gray.shape[::-1]}")
if mean_error / len(obj_points) < 1.0:
    print("Calibration quality: EXCELLENT (error < 1.0 pixel)")
elif mean_error / len(obj_points) < 2.0:
    print("Calibration quality: GOOD (error < 2.0 pixels)")
else:
    print("Calibration quality: ACCEPTABLE (consider more images or better lighting)")

# Visualization of reprojection
print("\n--- Reprojection Visualization ---")
# Loop over the list of successfully calibrated image paths
for i, img_path in enumerate(calibrated_image_paths):
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Draw detected corners in green
    cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, img_points[i], True)

    # Project points back
    img_points_reproj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    
    # Draw reprojected points in red
    for p in img_points_reproj:
        cv2.circle(img, (int(p[0][0]), int(p[0][1])), 5, (0, 0, 255), -1)

    # Save reprojection images for report (first 2 images)
    if i < 2:
        cv2.imwrite(f'reprojection_error_{i+1}.png', img)
        print(f"Saved reprojection_error_{i+1}.png")

    cv2.imshow('Reprojection', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Save calibration results to file for report
with open('calibration_results.txt', 'w') as f:
    f.write("Camera Calibration Results\n")
    f.write("========================\n\n")
    f.write(f"Number of images used: {len(obj_points)}\n")
    f.write(f"Image size: {gray.shape[::-1]}\n")
    f.write(f"Chessboard size: {CHESSBOARD_SIZE}\n")
    f.write(f"Square size: {SQUARE_SIZE} mm\n\n")
    f.write("Intrinsic Matrix (K):\n")
    f.write(f"{mtx}\n\n")
    f.write("Distortion Coefficients:\n")
    f.write(f"{dist}\n\n")
    f.write(f"Average Reprojection Error: {mean_error / len(obj_points):.4f} pixels\n")
    
print("\nCalibration results saved to calibration_results.txt")