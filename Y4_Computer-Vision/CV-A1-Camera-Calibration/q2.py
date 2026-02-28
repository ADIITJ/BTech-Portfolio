import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preprocess_image(img):
    """Enhanced preprocessing for better stereo matching"""
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(enhanced, 5, 80, 80)
    
    # Apply gentle Gaussian blur
    blurred = cv2.GaussianBlur(filtered, (3, 3), 0.8)
    
    return blurred

def compute_sad(patch1, patch2):
    """Compute Sum of Absolute Differences"""
    return np.sum(np.abs(patch1.astype(np.float32) - patch2.astype(np.float32)))

def compute_ssd(patch1, patch2):
    """Compute Sum of Squared Differences"""
    return np.sum((patch1.astype(np.float32) - patch2.astype(np.float32)) ** 2)

def compute_ncc(patch1, patch2):
    """Compute Normalized Cross Correlation"""
    patch1_flat = patch1.astype(np.float32).flatten()
    patch2_flat = patch2.astype(np.float32).flatten()
    
    # Normalize patches (zero mean)
    patch1_norm = patch1_flat - np.mean(patch1_flat)
    patch2_norm = patch2_flat - np.mean(patch2_flat)
    
    # Compute correlation coefficient
    numerator = np.dot(patch1_norm, patch2_norm)
    denominator = np.sqrt(np.sum(patch1_norm**2) * np.sum(patch2_norm**2))
    
    if denominator == 0:
        return -1  # Return worst possible NCC value
    
    return numerator / denominator

def compute_disparity_manual(left, right, window_size=13, max_disparity=80, metric='sad'):
    """Manual disparity computation"""
    print(f"Computing disparity using {metric.upper()} with {window_size}x{window_size} windows...")
    print(f"Image size: {left.shape[1]}x{left.shape[0]}, Max disparity: {max_disparity}")
    
    h, w = left.shape
    disparity = np.zeros((h, w), dtype=np.float32)
    half_window = window_size // 2
    
    # Choose similarity function
    if metric == 'sad':
        similarity_func = compute_sad
        is_better = lambda new_cost, best_cost: new_cost < best_cost
        initial_best = float('inf')
    elif metric == 'ssd':
        similarity_func = compute_ssd
        is_better = lambda new_cost, best_cost: new_cost < best_cost
        initial_best = float('inf')
    elif metric == 'ncc':
        similarity_func = compute_ncc
        is_better = lambda new_corr, best_corr: new_corr > best_corr
        initial_best = -1
    
    for y in range(half_window, h - half_window):
        if y % 100 == 0:
            progress = 100 * y / (h - 2 * half_window)
            print(f"Processing row {y}/{h} ({progress:.1f}%)")
            
        for x in range(half_window + max_disparity, w - half_window):
            left_patch = left[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
            
            best_cost = initial_best
            best_disparity = 0
            
            # Search along epipolar line
            for d in range(min(max_disparity, x - half_window)):
                x_right = x - d
                right_patch = right[y-half_window:y+half_window+1, x_right-half_window:x_right+half_window+1]
                
                cost = similarity_func(left_patch, right_patch)
                if is_better(cost, best_cost):
                    best_cost = cost
                    best_disparity = d
            
            disparity[y, x] = best_disparity
    
    return disparity

def apply_median_filter(disparity, kernel_size=5):
    """Apply median filter to reduce noise"""
    return cv2.medianBlur(disparity.astype(np.uint8), kernel_size).astype(np.float32)

def save_disparity_visualization(disparity, title, filename):
    """Save disparity map with proper visualization"""
    plt.figure(figsize=(12, 8))
    plt.imshow(disparity, cmap='plasma')
    plt.colorbar(label='Disparity (pixels)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def save_depth_visualization(depth, title, filename, baseline_mm=40):
    """Save depth map with proper visualization"""
    plt.figure(figsize=(12, 8))
    # Clip extreme depth values for better visualization
    depth_clipped = np.clip(depth, 200, np.percentile(depth[depth > 0], 95))
    plt.imshow(depth_clipped, cmap='plasma')
    plt.colorbar(label='Depth (mm)')
    plt.title(f'{title} (Baseline: {baseline_mm}mm)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def main():
    print("=== Q2 Stereo Vision - Complete Analysis with Correct 40mm Baseline ===")
    
    # Load stereo images
    print("Loading stereo images...")
    left_img = cv2.imread('images/left.jpeg')
    right_img = cv2.imread('images/right.jpeg')
    
    if left_img is None or right_img is None:
        print("Error: Could not load stereo images!")
        return
    
    print(f"Original image sizes:")
    print(f"  Left: {left_img.shape}")
    print(f"  Right: {right_img.shape}")
    
    # Convert to grayscale
    print("Converting to grayscale...")
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Keep full resolution for better quality
    print("Keeping original size: (1600, 1200)")
    
    # Apply preprocessing
    print("Applying enhanced preprocessing...")
    left_processed = preprocess_image(left_gray)
    right_processed = preprocess_image(right_gray)
    
    # Save input images for report
    cv2.imwrite('stereo_left_input.png', left_processed)
    cv2.imwrite('stereo_right_input.png', right_processed)
    print("Saved input images")
    
    # Camera parameters - CORRECT VALUES
    focal_length = 2839.59  # Full resolution focal length
    baseline = 40  # mm (CORRECT baseline)
    print(f"Camera parameters: f={focal_length:.2f}, baseline={baseline}mm")
    
    # Processing parameters
    max_disparity = 80
    
    print(f"\n=== EXPERIMENT 1: Effect of Window Size ===")
    window_sizes = [9, 13, 17]
    
    for i, window_size in enumerate(window_sizes):
        print(f"\n({i+1}/{len(window_sizes)}) Testing window size: {window_size}x{window_size}")
        
        # Compute disparity
        disparity = compute_disparity_manual(left_processed, right_processed,
                                           window_size, max_disparity, 'sad')
        print(f"Computed disparity range: {disparity.min():.2f} - {disparity.max():.2f}")
        
        # Apply post-processing
        disparity_filtered = apply_median_filter(disparity, 5)
        
        # Save disparity visualization
        save_disparity_visualization(disparity_filtered,
                                   f'Disparity Map - {window_size}×{window_size} Window',
                                   f'disparity_window_{window_size}.png')
        
        # Compute and save depth map with CORRECT 40mm baseline
        epsilon = 0.1
        depth = (focal_length * baseline) / (disparity_filtered + epsilon)
        save_depth_visualization(depth,
                               f'Depth Map - {window_size}×{window_size} Window',
                               f'depth_window_{window_size}.png',
                               baseline)
    
    print(f"\n=== EXPERIMENT 2: Effect of Similarity Metrics ===")
    metrics = ['sad', 'ssd', 'ncc']
    window_size = 13  # Optimal size
    
    for i, metric in enumerate(metrics):
        print(f"\n({i+1}/{len(metrics)}) Testing similarity metric: {metric.upper()}")
        
        # Compute disparity
        disparity = compute_disparity_manual(left_processed, right_processed,
                                           window_size, max_disparity, metric)
        print(f"Computed disparity range: {disparity.min():.2f} - {disparity.max():.2f}")
        
        # Apply post-processing
        disparity_filtered = apply_median_filter(disparity, 5)
        
        # Save disparity visualization
        save_disparity_visualization(disparity_filtered,
                                   f'Disparity Map - {metric.upper()} Metric',
                                   f'disparity_metric_{metric}.png')
    
    print(f"\n=== EXPERIMENT 3: Baseline Analysis (with CORRECT 40mm baseline) ===")
    
    # Use the best disparity map (13x13 window, SAD metric)
    print("Using optimal disparity map (13x13 window, SAD)...")
    disparity_best = compute_disparity_manual(left_processed, right_processed,
                                            13, max_disparity, 'sad')
    disparity_best_filtered = apply_median_filter(disparity_best, 5)
    
    # For baseline analysis, we'll simulate different baselines
    # to show how baseline affects depth accuracy
    simulated_baselines = [20, 40, 60]  # mm - show effect around true 40mm baseline
    
    for baseline_sim in simulated_baselines:
        print(f"Computing depth for simulated {baseline_sim}mm baseline...")
        
        epsilon = 0.1
        depth = (focal_length * baseline_sim) / (disparity_best_filtered + epsilon)
        save_depth_visualization(depth,
                               f'Depth Map - Baseline Analysis',
                               f'depth_baseline_{baseline_sim}.png',
                               baseline_sim)
    
    # Create the main depth map with correct 40mm baseline
    print("Creating main depth map with CORRECT 40mm baseline...")
    depth_main = (focal_length * baseline) / (disparity_best_filtered + epsilon)
    save_depth_visualization(depth_main,
                           'Main Depth Map - 13×13 Window, 40mm Baseline',
                           'depth_map_main.png',
                           baseline)
    
    print("\n=== All Q2 Analysis Completed Successfully with Correct 40mm Baseline! ===")
    print("Generated files:")
    results = [
        'stereo_left_input.png', 'stereo_right_input.png',
        'disparity_window_9.png', 'disparity_window_13.png', 'disparity_window_17.png',
        'depth_window_9.png', 'depth_window_13.png', 'depth_window_17.png',
        'disparity_metric_sad.png', 'disparity_metric_ssd.png', 'disparity_metric_ncc.png',
        'depth_baseline_20.png', 'depth_baseline_40.png', 'depth_baseline_60.png',
        'depth_map_main.png'
    ]
    for result in results:
        print(f"  - {result}")

if __name__ == "__main__":
    main()
