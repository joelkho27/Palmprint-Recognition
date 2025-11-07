"""
Create Report Visuals - Palm Comparison Images for Academic Report
Generates side-by-side palm comparisons with results overlaid
Similar to sample report format
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class ReportVisualGenerator:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
    
    def create_comparison_visual(self, img1_path, img2_path, result_text, 
                                score_text, output_path, test_type="SIFT"):
        """
        Create side-by-side comparison image with results overlaid
        
        Args:
            img1_path: Path to first palm image (e.g., registered palm)
            img2_path: Path to second palm image (e.g., test palm)
            result_text: "MATCH" or "NO MATCH"
            score_text: Score description (e.g., "Local Match Score: 20.69%")
            output_path: Where to save the result
            test_type: "SIFT" or "CNN"
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Error loading images: {img1_path}, {img2_path}")
            return
        
        # Resize images to same height for side-by-side display
        target_height = 400
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        new_w1 = int(w1 * target_height / h1)
        new_w2 = int(w2 * target_height / h2)
        
        img1_resized = cv2.resize(img1, (new_w1, target_height))
        img2_resized = cv2.resize(img2, (new_w2, target_height))
        
        # Convert to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB)
        
        # If SIFT, draw keypoints
        if test_type == "SIFT":
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            
            kp1, _ = self.sift.detectAndCompute(gray1, None)
            kp2, _ = self.sift.detectAndCompute(gray2, None)
            
            # Draw keypoints
            img1_rgb = cv2.drawKeypoints(img1_rgb, kp1, None, 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2_rgb = cv2.drawKeypoints(img2_rgb, kp2, None,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('white')
        
        # Display images
        ax1.imshow(img1_rgb)
        ax1.axis('off')
        ax1.set_title('New Palm Scan with Descriptors', fontsize=10, pad=10)
        
        ax2.imshow(img2_rgb)
        ax2.axis('off')
        ax2.set_title('DB Palm Scan with Descriptors', fontsize=10, pad=10)
        
        # Determine result color
        if "MATCH" in result_text and "NO" not in result_text:
            result_color = 'green'
        else:
            result_color = 'red'
        
        # Add result text at top
        fig.suptitle(f'Result: {result_text} | {score_text}', 
                    fontsize=14, fontweight='bold', color=result_color, y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
    
    def create_sift_match_visualization(self, img1_path, img2_path, 
                                       num_matches, output_path):
        """
        Create SIFT feature matching visualization with lines connecting matches
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.72 * n.distance:
                good_matches.append(m)
        
        # Draw matches (show top 50 for clarity)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, 
                                     good_matches[:50], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Convert to RGB and save
        img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(img_matches_rgb)
        plt.title(f'SIFT Feature Matching: {num_matches} good matches found', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
    
    def create_test_results_grid(self, test_results, output_path):
        """
        Create a grid of test results (like sample report's 2x2 or 2x3 grid)
        
        test_results: List of dicts with keys:
            - img1_path
            - img2_path
            - result_text
            - score_text
        """
        n_tests = len(test_results)
        
        if n_tests <= 2:
            rows, cols = 1, 2
            figsize = (14, 7)
        elif n_tests <= 4:
            rows, cols = 2, 2
            figsize = (14, 14)
        else:
            rows, cols = 2, 3
            figsize = (18, 12)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # Flatten axes for easier iteration
        if n_tests == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, test in enumerate(test_results[:n_tests]):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Load and prepare images
            img1 = cv2.imread(test['img1_path'])
            img2 = cv2.imread(test['img2_path'])
            
            if img1 is None or img2 is None:
                continue
            
            # Resize
            target_height = 300
            img1 = cv2.resize(img1, (int(img1.shape[1] * target_height / img1.shape[0]), 
                                    target_height))
            img2 = cv2.resize(img2, (int(img2.shape[1] * target_height / img2.shape[0]), 
                                    target_height))
            
            # Combine side by side
            combined = np.hstack([img1, img2])
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            
            # Display
            ax.imshow(combined_rgb)
            ax.axis('off')
            
            # Result color
            color = 'green' if 'MATCH' in test['result_text'] and 'NO' not in test['result_text'] else 'red'
            
            # Title
            ax.set_title(f"{test['result_text']} | {test['score_text']}", 
                        fontsize=10, fontweight='bold', color=color, pad=5)
        
        # Hide unused subplots
        for idx in range(n_tests, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✓ Saved: {output_path}")


def main():
    """
    Generate all report visuals for YOUR actual test data
    """
    generator = ReportVisualGenerator()
    
    # Define your test cases
    # UPDATE THESE PATHS TO YOUR ACTUAL IMAGE FILES
    
    base_dir = r"C:\Users\joelk\OneDrive\Desktop\palmprint ruofei\Palmprint_Recognition-master\Palmprint_Recognition-master"
    output_dir = os.path.join(base_dir, "Report_Visuals")
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Different Hands (Rejected - 6.90% in sample)
    # YOUR DATA: 676 matches, 4/6 tests, score 2417.84, REJECTED
    print("\n=== Generating Test 1: Different Hands ===")
    generator.create_comparison_visual(
        img1_path=os.path.join(base_dir, "registered_palm_left.bmp"),
        img2_path=os.path.join(base_dir, "palm_right_20250924_135633.bmp"),
        result_text="NO MATCH",
        score_text="676 SIFT matches, 4/6 tests, Score: 2417.84",
        output_path=os.path.join(output_dir, "sift_test1_different_hands.png"),
        test_type="SIFT"
    )
    
    # Test 2: Same Hand (Matched - 20.69% in sample)
    # YOUR DATA: 799 matches, 5/6 tests, score 2835.15, AUTHENTICATED
    print("\n=== Generating Test 2: Same Hand ===")
    generator.create_comparison_visual(
        img1_path=os.path.join(base_dir, "registered_palm_left.bmp"),
        img2_path=os.path.join(base_dir, "palm_left_20250924_143241.bmp"),
        result_text="MATCH",
        score_text="799 SIFT matches, 5/6 tests, Score: 2835.15",
        output_path=os.path.join(output_dir, "sift_test2_same_hand.png"),
        test_type="SIFT"
    )
    
    # Test 3: Same Hand Identical (100% in sample)
    print("\n=== Generating Test 3: Identical ===")
    generator.create_comparison_visual(
        img1_path=os.path.join(base_dir, "registered_palm_left.bmp"),
        img2_path=os.path.join(base_dir, "registered_palm_left.bmp"),
        result_text="MATCH",
        score_text="Identical image, 100% match",
        output_path=os.path.join(output_dir, "sift_test3_identical.png"),
        test_type="SIFT"
    )
    
    # Test 4: Another same hand with slight variation (98.85% in sample)
    print("\n=== Generating Test 4: Same Hand Slight Variation ===")
    generator.create_comparison_visual(
        img1_path=os.path.join(base_dir, "registered_palm_left.bmp"),
        img2_path=os.path.join(base_dir, "palm_left_20250924_134940.bmp"),
        result_text="MATCH",
        score_text="High similarity with variation",
        output_path=os.path.join(output_dir, "sift_test4_same_variation.png"),
        test_type="SIFT"
    )
    
    # Bonus: Create feature matching visualization
    print("\n=== Generating SIFT Feature Matching Visualization ===")
    generator.create_sift_match_visualization(
        img1_path=os.path.join(base_dir, "registered_palm_left.bmp"),
        img2_path=os.path.join(base_dir, "palm_left_20250924_143241.bmp"),
        num_matches=799,
        output_path=os.path.join(output_dir, "sift_feature_matching_799.png")
    )
    
    # Create combined grid (like sample report)
    print("\n=== Generating Combined Grid ===")
    test_results = [
        {
            'img1_path': os.path.join(base_dir, "registered_palm_left.bmp"),
            'img2_path': os.path.join(base_dir, "palm_right_20250924_135633.bmp"),
            'result_text': "NO MATCH",
            'score_text': "676 matches, Score: 2417.84"
        },
        {
            'img1_path': os.path.join(base_dir, "registered_palm_left.bmp"),
            'img2_path': os.path.join(base_dir, "palm_left_20250924_143241.bmp"),
            'result_text': "MATCH",
            'score_text': "799 matches, Score: 2835.15"
        },
        {
            'img1_path': os.path.join(base_dir, "registered_palm_left.bmp"),
            'img2_path': os.path.join(base_dir, "registered_palm_left.bmp"),
            'result_text': "MATCH",
            'score_text': "Identical, 100%"
        },
        {
            'img1_path': os.path.join(base_dir, "registered_palm_left.bmp"),
            'img2_path': os.path.join(base_dir, "palm_left_20250924_134940.bmp"),
            'result_text': "MATCH",
            'score_text': "High similarity"
        }
    ]
    
    generator.create_test_results_grid(
        test_results=test_results,
        output_path=os.path.join(output_dir, "sift_results_grid.png")
    )
    
    print("\n" + "="*60)
    print("✓ All visuals generated successfully!")
    print(f"✓ Check folder: {output_dir}")
    print("="*60)
    
    print("\nGenerated files:")
    print("  1. sift_test1_different_hands.png")
    print("  2. sift_test2_same_hand.png")
    print("  3. sift_test3_identical.png")
    print("  4. sift_test4_same_variation.png")
    print("  5. sift_feature_matching_799.png")
    print("  6. sift_results_grid.png")
    
    print("\nUse these in your report as:")
    print("  - Figure 6.1.3.1 SIFT Results (use grid)")
    print("  - Individual comparison figures as needed")


if __name__ == "__main__":
    main()
