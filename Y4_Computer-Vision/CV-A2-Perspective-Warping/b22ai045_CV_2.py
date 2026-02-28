"""
CSL7360: COMPUTER VISION - ASSIGNMENT 2
Name: Atharva Date
Roll No: B22AI045

Complete implementation of all assignment requirements.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

class CVAssignment:
    def __init__(self, image_path):
        self.image_path = image_path
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load and resize image to 250x250
        self.original_image = cv2.imread(image_path)
        self.original_image = cv2.resize(self.original_image, (250, 250))
        self.original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Store transformed images
        self.transformed_images = {}
        self.transformed_grays = {}
        
    def question_1_1_preprocessing(self):
        """Question 1.1: Image Transformation and Preprocessing (2 Marks)"""
        print("Question 1.1: Preprocessing & Transformations")
        
        # Original image
        self.transformed_images['original'] = self.original_image.copy()
        self.transformed_grays['original'] = self.original_gray.copy()
        
        # A. Rotation: rotate by 6° clockwise
        height, width = self.original_image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -6, 1.0)  # Negative for clockwise
        rotated = cv2.warpAffine(self.original_image, rotation_matrix, (width, height))
        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        
        self.transformed_images['rotation'] = rotated
        self.transformed_grays['rotation'] = rotated_gray
        
        # B. Scale: scale by 1.2x
        scaled = cv2.resize(self.original_image, None, fx=1.2, fy=1.2)
        # Crop or pad to maintain 250x250
        if scaled.shape[0] > 250 or scaled.shape[1] > 250:
            # Crop center
            start_y = (scaled.shape[0] - 250) // 2
            start_x = (scaled.shape[1] - 250) // 2
            scaled = scaled[start_y:start_y+250, start_x:start_x+250]
        else:
            # Pad
            pad_y = (250 - scaled.shape[0]) // 2
            pad_x = (250 - scaled.shape[1]) // 2
            scaled = cv2.copyMakeBorder(scaled, pad_y, 250-scaled.shape[0]-pad_y, 
                                       pad_x, 250-scaled.shape[1]-pad_x, cv2.BORDER_CONSTANT)
        
        scaled_gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        self.transformed_images['scale'] = scaled
        self.transformed_grays['scale'] = scaled_gray
        
        # C. Noise: add Gaussian noise (mean = 0, σ = 0.03)
        noise = np.random.normal(0, 0.03, self.original_image.shape)
        noisy = self.original_image.astype(np.float64) / 255.0 + noise
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
        
        self.transformed_images['noise'] = noisy
        self.transformed_grays['noise'] = noisy_gray
        
        # Display all transformed images
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        titles = ['Original', 'Rotation (6° CW)', 'Scale (1.2x)', 'Gaussian Noise']
        images = [self.transformed_images['original'], self.transformed_images['rotation'], 
                 self.transformed_images['scale'], self.transformed_images['noise']]
        
        for i, (ax, title, img) in enumerate(zip(axes.flat, titles, images)):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/q1_1_transformations.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Image transformations completed and saved")
        
    def harris_corner_detection(self, image, k=0.04, threshold=0.01):
        """Harris Corner Detection implementation from scratch"""
        # Convert to float
        img = image.astype(np.float64)
        
        # Compute gradients using Sobel operators
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute products of derivatives
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        
        # Apply Gaussian smoothing
        sigma = 1.5
        Ixx = ndimage.gaussian_filter(Ixx, sigma)
        Ixy = ndimage.gaussian_filter(Ixy, sigma)
        Iyy = ndimage.gaussian_filter(Iyy, sigma)
        
        # Compute Harris response
        det_M = Ixx * Iyy - Ixy * Ixy
        trace_M = Ixx + Iyy
        R = det_M - k * (trace_M ** 2)
        
        # Find corners
        corners = R > threshold * R.max()
        
        # Non-maximum suppression
        corner_coords = []
        for y in range(1, R.shape[0]-1):
            for x in range(1, R.shape[1]-1):
                if corners[y, x] and R[y, x] == R[y-1:y+2, x-1:x+2].max():
                    corner_coords.append((x, y))
        
        return corner_coords, R
    
    def question_1_2_harris_corners(self):
        """Question 1.2: Harris Corner Detection (3 Marks)"""
        print("\nQuestion 1.2: Harris Corner Detection")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        corner_counts = {}
        all_corners = {}
        
        for i, (name, img_gray) in enumerate(self.transformed_grays.items()):
            # Detect Harris corners
            corners, response = self.harris_corner_detection(img_gray)
            corner_counts[name] = len(corners)
            all_corners[name] = corners
            
            # Display original image
            axes[0, i].imshow(img_gray, cmap='gray')
            axes[0, i].set_title(f'{name.title()} Image', fontsize=12)
            axes[0, i].axis('off')
            
            # Display corners
            img_with_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            for x, y in corners:
                cv2.circle(img_with_corners, (x, y), 2, (255, 0, 0), -1)
            
            axes[1, i].imshow(img_with_corners)
            axes[1, i].set_title(f'Harris Corners ({len(corners)} detected)', fontsize=12)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/q1_2_harris_corners.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print comparison
        print("Corner Detection Results:")
        for name, count in corner_counts.items():
            print(f"  {name.title()}: {count} corners detected")
        
        self.harris_corners = all_corners
        print("✓ Harris corner detection completed")
        
    def build_gaussian_pyramid(self, image, num_octaves=3, scales_per_octave=3):
        """Build Gaussian pyramid for DoG computation"""
        pyramid = []
        sigma = 1.6
        k = 2 ** (1.0 / scales_per_octave)
        
        current_img = image.astype(np.float64)
        
        for octave in range(num_octaves):
            octave_imgs = []
            for scale in range(scales_per_octave + 3):  # +3 for DoG computation
                if octave == 0 and scale == 0:
                    # First image
                    octave_imgs.append(current_img)
                else:
                    # Apply Gaussian blur
                    current_sigma = sigma * (k ** scale)
                    if octave > 0 and scale == 0:
                        # Downsample from previous octave
                        prev_octave = pyramid[octave-1]
                        current_img = cv2.resize(prev_octave[scales_per_octave], 
                                               (current_img.shape[1]//2, current_img.shape[0]//2))
                    
                    blurred = ndimage.gaussian_filter(current_img, current_sigma)
                    octave_imgs.append(blurred)
            
            pyramid.append(octave_imgs)
            current_img = octave_imgs[0]
        
        return pyramid
    
    def compute_dog_pyramid(self, gaussian_pyramid):
        """Compute Difference of Gaussian pyramid"""
        dog_pyramid = []
        
        for octave in gaussian_pyramid:
            dog_octave = []
            for i in range(len(octave) - 1):
                dog = octave[i+1] - octave[i]
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)
        
        return dog_pyramid
    
    def normalize_dog_responses(self, dog_pyramid):
        """Normalize DoG responses per octave"""
        normalized_pyramid = []
        
        for dog_octave in dog_pyramid:
            normalized_octave = []
            for dog_img in dog_octave:
                # Normalize to [0, 1]
                normalized = (dog_img - dog_img.min()) / (dog_img.max() - dog_img.min() + 1e-8)
                normalized_octave.append(normalized)
            normalized_pyramid.append(normalized_octave)
        
        return normalized_pyramid
    
    def detect_keypoints(self, dog_pyramid, contrast_threshold=0.03):
        """Detect keypoints based on local extrema in DoG"""
        keypoints = []
        
        for octave_idx, dog_octave in enumerate(dog_pyramid):
            for scale_idx in range(1, len(dog_octave) - 1):  # Skip first and last scales
                current = dog_octave[scale_idx]
                prev_scale = dog_octave[scale_idx - 1]
                next_scale = dog_octave[scale_idx + 1]
                
                h, w = current.shape
                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        val = current[y, x]
                        
                        # Check if it's an extremum
                        if abs(val) < contrast_threshold:
                            continue
                        
                        # 3x3x3 neighborhood check
                        neighborhood = np.array([
                            prev_scale[y-1:y+2, x-1:x+2],
                            current[y-1:y+2, x-1:x+2], 
                            next_scale[y-1:y+2, x-1:x+2]
                        ])
                        
                        is_max = val == neighborhood.max()
                        is_min = val == neighborhood.min()
                        
                        if is_max or is_min:
                            # Scale coordinates back to original image size
                            scale_factor = 2 ** octave_idx
                            keypoints.append((x * scale_factor, y * scale_factor, 
                                            octave_idx, scale_idx, val))
        
        return keypoints
    
    def compute_orientation_histogram(self, image, x, y, radius=8, bins=8):
        """Compute orientation histogram for a patch"""
        # Compute gradients
        gx = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx*gx + gy*gy)
        orientation = np.arctan2(gy, gx) * 180 / np.pi
        orientation[orientation < 0] += 360
        
        hist = np.zeros(bins)
        bin_width = 360.0 / bins
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    mag = magnitude[ny, nx]
                    ori = orientation[ny, nx]
                    bin_idx = int(ori / bin_width) % bins
                    hist[bin_idx] += mag
        
        return hist
    
    def compute_sift_descriptor(self, image, keypoints, patch_size=16):
        """Compute SIFT-like 128-dimensional descriptors"""
        descriptors = []
        
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            
            # Check bounds
            half_size = patch_size // 2
            if (x - half_size < 0 or x + half_size >= image.shape[1] or 
                y - half_size < 0 or y + half_size >= image.shape[0]):
                continue
            
            # Extract patch
            patch = image[y-half_size:y+half_size, x-half_size:x+half_size]
            
            # Divide into 4x4 cells
            descriptor = []
            cell_size = patch_size // 4
            
            for cy in range(4):
                for cx in range(4):
                    cell = patch[cy*cell_size:(cy+1)*cell_size, 
                               cx*cell_size:(cx+1)*cell_size]
                    
                    # Compute orientation histogram for this cell
                    center_x, center_y = cell_size // 2, cell_size // 2
                    hist = self.compute_orientation_histogram(cell, center_x, center_y, 
                                                            radius=cell_size//2, bins=8)
                    descriptor.extend(hist)
            
            # Normalize descriptor
            descriptor = np.array(descriptor)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
            
            descriptors.append(descriptor)
        
        return np.array(descriptors)
    
    def question_2_1_keypoint_detection(self):
        """Question 2.1: Keypoint Detection & Descriptor Computation (4 Marks)"""
        print("\nQuestion 2.1: Keypoint Detection & Descriptor Computation")
        
        self.keypoints = {}
        self.descriptors = {}
        
        for name, img_gray in self.transformed_grays.items():
            print(f"Processing {name} image...")
            
            # Build Gaussian pyramid
            gauss_pyramid = self.build_gaussian_pyramid(img_gray)
            
            # Compute DoG pyramid
            dog_pyramid = self.compute_dog_pyramid(gauss_pyramid)
            
            # Normalize DoG responses
            normalized_dog = self.normalize_dog_responses(dog_pyramid)
            
            # Detect keypoints
            keypoints = self.detect_keypoints(normalized_dog)
            
            # Compute descriptors
            valid_keypoints = [(kp[0], kp[1]) for kp in keypoints]
            descriptors = self.compute_sift_descriptor(img_gray, keypoints)
            
            self.keypoints[name] = valid_keypoints[:len(descriptors)]  # Match lengths
            self.descriptors[name] = descriptors
            
            print(f"  Detected {len(self.keypoints[name])} keypoints")
        
        # Visualize keypoints
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for i, (name, img_gray) in enumerate(self.transformed_grays.items()):
            img_with_kp = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            
            for x, y in self.keypoints[name]:
                cv2.circle(img_with_kp, (int(x), int(y)), 2, (0, 255, 0), -1)
            
            axes[i].imshow(img_with_kp)
            axes[i].set_title(f'{name.title()}\n({len(self.keypoints[name])} keypoints)', fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/q2_1_keypoints.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Keypoint detection and descriptor computation completed")
    
    def match_descriptors(self, desc1, desc2, ratio_threshold=0.75):
        """Match descriptors using Euclidean distance and Lowe's ratio test"""
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Compute pairwise distances
        distances = cdist(desc1, desc2, metric='euclidean')
        
        matches = []
        for i in range(len(desc1)):
            # Find two nearest neighbors
            sorted_indices = np.argsort(distances[i])
            
            if len(sorted_indices) >= 2:
                nearest_dist = distances[i, sorted_indices[0]]
                second_nearest_dist = distances[i, sorted_indices[1]]
                
                # Lowe's ratio test
                if nearest_dist < ratio_threshold * second_nearest_dist:
                    matches.append((i, sorted_indices[0], nearest_dist))
        
        return matches
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, title="Matches"):
        """Visualize descriptor matches"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create combined image
        combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        combined[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # Draw matches
        for match in matches[:50]:  # Limit to first 50 matches for clarity
            idx1, idx2, _ = match
            if idx1 < len(kp1) and idx2 < len(kp2):
                pt1 = (int(kp1[idx1][0]), int(kp1[idx1][1]))
                pt2 = (int(kp2[idx2][0] + w1), int(kp2[idx2][1]))
                
                cv2.circle(combined, pt1, 3, (0, 255, 0), -1)
                cv2.circle(combined, pt2, 3, (0, 255, 0), -1)
                cv2.line(combined, pt1, pt2, (255, 0, 0), 1)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(combined)
        plt.title(f'{title} ({len(matches)} matches)')
        plt.axis('off')
        plt.tight_layout()
        
        return combined
    
    def question_2_2_descriptor_matching(self):
        """Question 2.2: Descriptor Matching (3 Marks)"""
        print("\nQuestion 2.2: Descriptor Matching")
        
        self.matches = {}
        original_desc = self.descriptors['original']
        original_kp = self.keypoints['original']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        
        transform_names = ['rotation', 'scale', 'noise']
        for i, name in enumerate(transform_names):
            print(f"Matching {name} with original...")
            
            # Match descriptors
            matches = self.match_descriptors(original_desc, self.descriptors[name])
            self.matches[name] = matches
            
            # Visualize matches
            combined_img = self.visualize_matches(
                self.transformed_grays['original'], 
                self.transformed_grays[name],
                original_kp, 
                self.keypoints[name], 
                matches,
                f'Original vs {name.title()}'
            )
            
            axes[i].imshow(combined_img)
            axes[i].set_title(f'Original vs {name.title()} ({len(matches)} matches)', fontsize=14)
            axes[i].axis('off')
            
            print(f"  Found {len(matches)} matches")
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/q2_2_matches.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Descriptor matching completed")
    
    def normalize_points(self, points):
        """Normalize points for numerical stability"""
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        
        # Translate points to origin
        translated = points - centroid
        
        # Scale points so average distance from origin is sqrt(2)
        distances = np.sqrt(np.sum(translated**2, axis=1))
        avg_distance = np.mean(distances)
        
        if avg_distance > 0:
            scale = np.sqrt(2) / avg_distance
        else:
            scale = 1.0
        
        # Normalization matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        
        # Apply normalization
        normalized = scale * translated
        
        return normalized, T
    
    def question_3_1_point_correspondence(self):
        """Question 3.1: Point Correspondence & Normalization (2 Marks)"""
        print("\nQuestion 3.1: Point Correspondence & Normalization")
        
        self.point_correspondences = {}
        self.normalized_points = {}
        
        original_kp = np.array(self.keypoints['original'])
        
        for name in ['rotation', 'scale', 'noise']:
            matches = self.matches[name]
            transform_kp = np.array(self.keypoints[name])
            
            # Extract matched points
            points1 = []
            points2 = []
            
            for match in matches:
                idx1, idx2, _ = match
                if idx1 < len(original_kp) and idx2 < len(transform_kp):
                    points1.append(original_kp[idx1])
                    points2.append(transform_kp[idx2])
            
            points1 = np.array(points1)
            points2 = np.array(points2)
            
            # Normalize points
            norm_points1, T1 = self.normalize_points(points1)
            norm_points2, T2 = self.normalize_points(points2)
            
            self.point_correspondences[name] = (points1, points2)
            self.normalized_points[name] = (norm_points1, norm_points2, T1, T2)
            
            print(f"{name}: {len(points1)} point correspondences extracted and normalized")
        
        print("✓ Point correspondence and normalization completed")
    
    def eight_point_algorithm(self, points1, points2):
        """Implement 8-point algorithm for fundamental matrix estimation"""
        n = len(points1)
        
        # Construct coefficient matrix A
        A = np.zeros((n, 9))
        for i in range(n):
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        
        # Solve Af = 0 using SVD
        U, S, Vt = np.linalg.svd(A)
        f = Vt[-1]  # Last row of V (smallest singular value)
        
        # Reshape to 3x3 matrix
        F = f.reshape(3, 3)
        
        return F
    
    def enforce_rank2_constraint(self, F):
        """Apply rank-2 constraint on fundamental matrix"""
        U, S, Vt = np.linalg.svd(F)
        S[-1] = 0  # Set smallest singular value to 0
        F_corrected = U @ np.diag(S) @ Vt
        return F_corrected
    
    def question_3_2_fundamental_matrix(self):
        """Question 3.2: Fundamental Matrix Estimation (3 Marks)"""
        print("\nQuestion 3.2: Fundamental Matrix Estimation")
        
        self.fundamental_matrices = {}
        
        for name in ['rotation', 'scale', 'noise']:
            norm_points1, norm_points2, T1, T2 = self.normalized_points[name]
            
            if len(norm_points1) >= 8:
                # Apply 8-point algorithm on normalized points
                F_normalized = self.eight_point_algorithm(norm_points1, norm_points2)
                
                # Enforce rank-2 constraint
                F_normalized = self.enforce_rank2_constraint(F_normalized)
                
                # Denormalize fundamental matrix
                F = T2.T @ F_normalized @ T1
                
                self.fundamental_matrices[name] = F
                
                print(f"{name}: Fundamental matrix computed")
                print(f"  Rank: {np.linalg.matrix_rank(F)}")
                print(f"  Condition number: {np.linalg.cond(F):.2e}")
            else:
                print(f"{name}: Insufficient points for 8-point algorithm")
        
        print("✓ Fundamental matrix estimation completed")
    
    def sampson_distance(self, F, points1, points2):
        """Compute Sampson distance for point correspondences"""
        distances = []
        
        for p1, p2 in zip(points1, points2):
            x1 = np.array([p1[0], p1[1], 1])
            x2 = np.array([p2[0], p2[1], 1])
            
            # Epipolar line in second image
            l2 = F @ x1
            # Epipolar line in first image  
            l1 = F.T @ x2
            
            # Sampson distance
            numerator = (x2 @ F @ x1) ** 2
            denominator = l1[0]**2 + l1[1]**2 + l2[0]**2 + l2[1]**2
            
            if denominator > 0:
                distance = numerator / denominator
            else:
                distance = float('inf')
            
            distances.append(distance)
        
        return np.array(distances)
    
    def ransac_fundamental_matrix(self, points1, points2, num_iterations=2000, threshold=1.0):
        """RANSAC implementation for robust fundamental matrix estimation"""
        best_F = None
        best_inliers = []
        max_inliers = 0
        
        n_points = len(points1)
        
        for iteration in range(num_iterations):
            # Randomly sample 8 correspondences
            if n_points < 8:
                break
                
            indices = np.random.choice(n_points, 8, replace=False)
            sample_points1 = points1[indices]
            sample_points2 = points2[indices]
            
            # Normalize sample points
            norm_p1, T1 = self.normalize_points(sample_points1)
            norm_p2, T2 = self.normalize_points(sample_points2)
            
            # Compute fundamental matrix using 8-point algorithm
            try:
                F_norm = self.eight_point_algorithm(norm_p1, norm_p2)
                F_norm = self.enforce_rank2_constraint(F_norm)
                F = T2.T @ F_norm @ T1
                
                # Count inliers using Sampson distance
                distances = self.sampson_distance(F, points1, points2)
                inliers = np.where(distances < threshold)[0]
                
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_F = F
                    best_inliers = inliers
                    
            except np.linalg.LinAlgError:
                continue
        
        return best_F, best_inliers
    
    def draw_epipolar_lines(self, img1, img2, points1, points2, F, inliers):
        """Draw epipolar lines for inlier correspondences"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create images for drawing
        img1_lines = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2_lines = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        
        # Draw epipolar lines for inliers only
        for idx in inliers[:20]:  # Limit to first 20 for clarity
            pt1 = points1[idx]
            pt2 = points2[idx]
            
            # Compute epipolar line in second image
            x1_homo = np.array([pt1[0], pt1[1], 1])
            line2 = F @ x1_homo
            
            # Compute epipolar line in first image
            x2_homo = np.array([pt2[0], pt2[1], 1])
            line1 = F.T @ x2_homo
            
            # Draw lines
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Line in image 2
            if abs(line2[1]) > 1e-6:  # Not vertical
                y_start, y_end = 0, h2-1
                x_start = int(-(line2[2] + line2[1]*y_start) / line2[0]) if abs(line2[0]) > 1e-6 else 0
                x_end = int(-(line2[2] + line2[1]*y_end) / line2[0]) if abs(line2[0]) > 1e-6 else w2-1
                x_start, x_end = np.clip([x_start, x_end], 0, w2-1)
                cv2.line(img2_lines, (x_start, y_start), (x_end, y_end), color, 1)
            
            # Line in image 1  
            if abs(line1[1]) > 1e-6:  # Not vertical
                y_start, y_end = 0, h1-1
                x_start = int(-(line1[2] + line1[1]*y_start) / line1[0]) if abs(line1[0]) > 1e-6 else 0
                x_end = int(-(line1[2] + line1[1]*y_end) / line1[0]) if abs(line1[0]) > 1e-6 else w1-1
                x_start, x_end = np.clip([x_start, x_end], 0, w1-1)
                cv2.line(img1_lines, (x_start, y_start), (x_end, y_end), color, 1)
            
            # Draw corresponding points
            cv2.circle(img1_lines, (int(pt1[0]), int(pt1[1])), 3, color, -1)
            cv2.circle(img2_lines, (int(pt2[0]), int(pt2[1])), 3, color, -1)
        
        return img1_lines, img2_lines
    
    def question_3_3_ransac_fundamental(self):
        """Question 3.3: RANSAC for Robust Fundamental Matrix (3 Marks)"""
        print("\nQuestion 3.3: RANSAC for Robust Fundamental Matrix")
        
        self.ransac_results = {}
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        for i, name in enumerate(['rotation', 'scale', 'noise']):
            points1, points2 = self.point_correspondences[name]
            
            print(f"Running RANSAC for {name}...")
            
            # Run RANSAC
            F_robust, inliers = self.ransac_fundamental_matrix(
                points1, points2, num_iterations=2000, threshold=1.0
            )
            
            self.ransac_results[name] = {
                'fundamental_matrix': F_robust,
                'inliers': inliers,
                'total_matches': len(points1),
                'inlier_ratio': len(inliers) / len(points1) if len(points1) > 0 else 0
            }
            
            print(f"  Total matches: {len(points1)}")
            print(f"  Inliers: {len(inliers)}")
            print(f"  Inlier ratio: {len(inliers)/len(points1):.3f}")
            
            # Draw epipolar lines
            if F_robust is not None:
                img1_epi, img2_epi = self.draw_epipolar_lines(
                    self.transformed_grays['original'],
                    self.transformed_grays[name],
                    points1, points2, F_robust, inliers
                )
                
                axes[i, 0].imshow(img1_epi)
                axes[i, 0].set_title(f'Original Image - {name.title()}\nEpipolar Lines ({len(inliers)} inliers)', fontsize=12)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(img2_epi)
                axes[i, 1].set_title(f'{name.title()} Image\nEpipolar Lines ({len(inliers)} inliers)', fontsize=12)
                axes[i, 1].axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/q3_3_epipolar_lines.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ RANSAC fundamental matrix estimation completed")
    
    def run_complete_assignment(self):
        """Run the complete assignment"""
        print("="*60)
        print("CSL7360: COMPUTER VISION - ASSIGNMENT 2")
        print("Name: Atharva Date | Roll No: B22AI045")
        print("="*60)
        
        # Question 1.1
        self.question_1_1_preprocessing()
        
        # Question 1.2  
        self.question_1_2_harris_corners()
        
        # Question 2.1
        self.question_2_1_keypoint_detection()
        
        # Question 2.2
        self.question_2_2_descriptor_matching()
        
        # Question 3.1
        self.question_3_1_point_correspondence()
        
        # Question 3.2
        self.question_3_2_fundamental_matrix()
        
        # Question 3.3
        self.question_3_3_ransac_fundamental()
        
        print("\n" + "="*60)
        print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
        print("All results saved in 'results' directory")
        print("="*60)

if __name__ == "__main__":
    # Initialize and run assignment
    cv_assignment = CVAssignment("Milan_Cathedral.jpg")
    cv_assignment.run_complete_assignment()