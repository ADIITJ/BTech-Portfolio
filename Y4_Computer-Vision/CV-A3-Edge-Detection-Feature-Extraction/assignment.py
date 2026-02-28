import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.stats import multivariate_normal
import maxflow

print("="*60)
print("CSL7360: Computer Vision - Assignment 3")
print("Student: Atharva Date")
print("Roll Number: B22AI045")
print("="*60)

# ============================================================================
# QUESTION 1: LUCAS-KANADE OPTICAL FLOW
# ============================================================================

class HarrisCornerDetector:
    def __init__(self, k=0.04, threshold=0.01, window_size=3):
        self.k = k
        self.threshold = threshold
        self.window_size = window_size
    
    def detect(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32)
        
        Ix = convolve(gray, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        Iy = convolve(gray, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
        
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        window = np.ones((self.window_size, self.window_size))
        Sxx = convolve(Ixx, window)
        Syy = convolve(Iyy, window)
        Sxy = convolve(Ixy, window)
        
        det_M = (Sxx * Syy) - (Sxy ** 2)
        trace_M = Sxx + Syy
        R = det_M - self.k * (trace_M ** 2)
        
        corner_threshold = self.threshold * R.max()
        corners = R > corner_threshold
        
        corner_coords = np.argwhere(corners)
        corner_coords = corner_coords[:, [1, 0]]
        
        return corner_coords, R

class LucasKanadeOpticalFlow:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.half_window = window_size // 2
    
    def compute_flow(self, img1, img2, points):
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
        
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        gray1 = gray1.astype(np.float32)
        gray2 = gray2.astype(np.float32)
        
        Ix = convolve(gray1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])) / 8.0
        Iy = convolve(gray1, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])) / 8.0
        It = gray2 - gray1
        
        flows = []
        valid_points = []
        
        for point in points:
            x, y = int(point[0]), int(point[1])
            
            if (x - self.half_window < 0 or x + self.half_window >= gray1.shape[1] or
                y - self.half_window < 0 or y + self.half_window >= gray1.shape[0]):
                continue
            
            patch_Ix = Ix[y - self.half_window:y + self.half_window + 1,
                          x - self.half_window:x + self.half_window + 1].flatten()
            patch_Iy = Iy[y - self.half_window:y + self.half_window + 1,
                          x - self.half_window:x + self.half_window + 1].flatten()
            patch_It = It[y - self.half_window:y + self.half_window + 1,
                          x - self.half_window:x + self.half_window + 1].flatten()
            
            A = np.vstack((patch_Ix, patch_Iy)).T
            b = -patch_It
            
            ATA = A.T @ A
            
            if np.linalg.det(ATA) < 1e-5:
                continue
            
            flow = np.linalg.lstsq(A, b, rcond=None)[0]
            
            flows.append(flow)
            valid_points.append(point)
        
        return np.array(valid_points), np.array(flows)

def visualize_optical_flow(img1, img2, points, flows, output_path):
    result = img2.copy()
    
    for i, (point, flow) in enumerate(zip(points, flows)):
        x, y = int(point[0]), int(point[1])
        fx, fy = flow
        
        cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
        
        end_x = int(x + fx)
        end_y = int(y + fy)
        cv2.arrowedLine(result, (x, y), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)
    
    cv2.imwrite(output_path, result)
    return result

def run_question1():
    print("\n" + "="*60)
    print("QUESTION 1: Lucas-Kanade Optical Flow")
    print("="*60)
    
    img1 = cv2.imread('images/frame10.png')
    img2 = cv2.imread('images/frame11.png')
    
    if img1 is None or img2 is None:
        print("ERROR: Please place frame10.png and frame11.png in the images/ directory")
        return
    
    cv2.imwrite('outputs/frame10.png', img1)
    cv2.imwrite('outputs/frame11.png', img2)
    
    harris = HarrisCornerDetector(k=0.04, threshold=0.01, window_size=3)
    corners, response = harris.detect(img1)
    
    corners = corners[::10]
    
    lk = LucasKanadeOpticalFlow(window_size=15)
    valid_points, flows = lk.compute_flow(img1, img2, corners)
    
    print(f"✓ Detected {len(corners)} corners")
    print(f"✓ Successfully tracked {len(valid_points)} points")
    
    result = visualize_optical_flow(img1, img2, valid_points, flows, 
                                    'outputs/optical_flow_result.jpg')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Frame 1')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Frame 2')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Optical Flow Visualization')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/optical_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    corner_img = img1.copy()
    for corner in corners:
        cv2.circle(corner_img, (int(corner[0]), int(corner[1])), 3, (0, 0, 255), -1)
    cv2.imwrite('outputs/harris_corners.jpg', corner_img)
    
    print("✓ Results saved to outputs/")
    print("  - harris_corners.jpg")
    print("  - optical_flow_result.jpg")
    print("  - optical_flow_comparison.png")

# ============================================================================
# QUESTION 2: GRAPH CUT SEGMENTATION
# ============================================================================

class GraphCutSegmentation:
    def __init__(self, image, smoothness_weight=10.0):
        self.image = image.astype(np.float32)
        self.smoothness_weight = smoothness_weight
        self.height, self.width = image.shape[:2]
        self.fg_model = None
        self.bg_model = None
    
    def fit_gaussian_models(self, fg_seeds, bg_seeds):
        fg_pixels = self.image[fg_seeds[:, 1], fg_seeds[:, 0]]
        bg_pixels = self.image[bg_seeds[:, 1], bg_seeds[:, 0]]
        
        self.fg_mean = np.mean(fg_pixels, axis=0)
        self.fg_cov = np.cov(fg_pixels.T) + np.eye(3) * 1e-5
        
        self.bg_mean = np.mean(bg_pixels, axis=0)
        self.bg_cov = np.cov(bg_pixels.T) + np.eye(3) * 1e-5
    
    def compute_data_term(self, pixel):
        fg_prob = multivariate_normal.pdf(pixel, self.fg_mean, self.fg_cov)
        bg_prob = multivariate_normal.pdf(pixel, self.bg_mean, self.bg_cov)
        
        fg_cost = -np.log(fg_prob + 1e-10)
        bg_cost = -np.log(bg_prob + 1e-10)
        
        return fg_cost, bg_cost
    
    def compute_smoothness_term(self, pixel1, pixel2):
        diff = np.linalg.norm(pixel1 - pixel2)
        return self.smoothness_weight * np.exp(-diff / 10.0)
    
    def segment(self, fg_seeds, bg_seeds):
        self.fit_gaussian_models(fg_seeds, bg_seeds)
        
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                pixel = self.image[y, x]
                fg_cost, bg_cost = self.compute_data_term(pixel)
                
                g.add_tedge(nodeids[y, x], fg_cost, bg_cost)
        
        for y in range(self.height):
            for x in range(self.width):
                if x < self.width - 1:
                    weight = self.compute_smoothness_term(
                        self.image[y, x], self.image[y, x + 1])
                    g.add_edge(nodeids[y, x], nodeids[y, x + 1], weight, weight)
                
                if y < self.height - 1:
                    weight = self.compute_smoothness_term(
                        self.image[y, x], self.image[y + 1, x])
                    g.add_edge(nodeids[y, x], nodeids[y + 1, x], weight, weight)
        
        g.maxflow()
        
        segmentation = g.get_grid_segments(nodeids)
        
        return segmentation.astype(np.uint8) * 255

def visualize_seeds(image, fg_seeds, bg_seeds, output_path):
    vis_img = image.copy()
    
    for seed in fg_seeds:
        cv2.circle(vis_img, tuple(seed), 5, (0, 255, 0), -1)
    
    for seed in bg_seeds:
        cv2.circle(vis_img, tuple(seed), 5, (0, 0, 255), -1)
    
    cv2.imwrite(output_path, vis_img)

def experiment_smoothness_weights(image, fg_seeds, bg_seeds, image_name):
    weights = [1.0, 10.0, 50.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, weight in enumerate(weights):
        gc = GraphCutSegmentation(image, smoothness_weight=weight)
        segmentation = gc.segment(fg_seeds, bg_seeds)
        
        axes[0, idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, idx].set_title(f'Original Image')
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(segmentation, cmap='gray')
        axes[1, idx].set_title(f'Smoothness λ={weight}')
        axes[1, idx].axis('off')
        
        cv2.imwrite(f'outputs/{image_name}_segmentation_weight_{weight}.jpg', segmentation)
    
    plt.tight_layout()
    plt.savefig(f'outputs/{image_name}_smoothness_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def run_question2():
    print("\n" + "="*60)
    print("QUESTION 2: Graph Cut Segmentation")
    print("="*60)
    
    img1 = cv2.imread('images/bird.jpg')
    
    if img1 is None:
        print("ERROR: bird.jpg not found in images/ directory")
        return
    
    cv2.imwrite('outputs/bird_original.jpg', img1)
    
    h, w = img1.shape[:2]
    
    fg_seeds1 = np.array([
        [w//2, h//2], [w//2 + 20, h//2], [w//2 - 20, h//2],
        [w//2, h//2 + 20], [w//2, h//2 - 20],
        [w//2 + 15, h//2 + 15], [w//2 - 15, h//2 - 15]
    ])
    
    bg_seeds1 = np.array([
        [30, 30], [w-30, 30], [30, h-30], [w-30, h-30],
        [w//2, 30], [w//2, h-30], [30, h//2], [w-30, h//2],
        [50, 50], [w-50, 50], [50, h-50], [w-50, h-50]
    ])
    
    visualize_seeds(img1, fg_seeds1, bg_seeds1, 'outputs/bird_seeds.jpg')
    
    print("✓ Processing bird image with different smoothness weights...")
    experiment_smoothness_weights(img1, fg_seeds1, bg_seeds1, 'bird')
    
    gc1 = GraphCutSegmentation(img1, smoothness_weight=10.0)
    seg1 = gc1.segment(fg_seeds1, bg_seeds1)
    
    result1 = img1.copy()
    result1[seg1 == 0] = result1[seg1 == 0] * 0.3
    cv2.imwrite('outputs/bird_final_overlay.jpg', result1)
    
    print("✓ Results saved to outputs/")
    print("  - bird_seeds.jpg")
    print("  - bird_segmentation_weight_1.0.jpg")
    print("  - bird_segmentation_weight_10.0.jpg")
    print("  - bird_segmentation_weight_50.0.jpg")
    print("  - bird_smoothness_comparison.png")
    print("  - bird_final_overlay.jpg")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_question1()
    run_question2()
    
    print("\n" + "="*60)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All results saved to outputs/")
    print("\nTo compile the report:")
    print("  pdflatex report.tex")
    print("  pdflatex report.tex")
    print("="*60)
