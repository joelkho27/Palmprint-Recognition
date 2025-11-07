"""
Comprehensive SIFT Testing on Tongji Palmprint Dataset
Calculates GAR, FAR, FRR, Accuracy and generates report visuals
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import your SIFT functions
from SIFT_DIP import sift_detect_match_num, get_sift_features


class DatasetTester:
    def __init__(self, testing_dir, ratio_threshold=0.75, match_threshold=3):
        self.testing_dir = testing_dir
        self.ratio_threshold = ratio_threshold
        self.match_threshold = match_threshold
        self.results = {
            'genuine_tests': [],  # Same person
            'impostor_tests': [],  # Different person
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline matching your system"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Bilateral filtering
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive gamma correction
        mean_val = np.mean(enhanced)
        if mean_val < 100:
            gamma = 1.2
        elif mean_val > 150:
            gamma = 0.8
        else:
            gamma = 1.0
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def sift_match_safe(self, des1, des2):
        """Safe SIFT matching with error handling"""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
        
        try:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            match_num = 0
            
            for match_pair in matches:
                if len(match_pair) == 2:  # Only process if we have 2 matches
                    first, second = match_pair
                    if first.distance < self.ratio_threshold * second.distance:
                        match_num += 1
            
            return match_num
        except Exception as e:
            return 0
    
    def load_all_images(self):
        """Load all images from dataset with SIFT features"""
        print("Loading dataset images and extracting SIFT features...")
        images = {}
        
        files = sorted([f for f in os.listdir(self.testing_dir) if f.endswith('.bmp')])
        
        for filename in tqdm(files, desc="Loading images"):
            person_id = filename[:3]  # e.g., "001"
            sample_id = filename[4]    # e.g., "4", "5", "6"
            
            if person_id not in images:
                images[person_id] = {}
            
            img_path = os.path.join(self.testing_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                preprocessed = self.preprocess_image(img)
                kp, des = get_sift_features(preprocessed, dect_type='sift')
                
                images[person_id][sample_id] = {
                    'path': img_path,
                    'image': preprocessed,
                    'descriptors': des,
                    'filename': filename
                }
        
        print(f"✓ Loaded {len(images)} people, {sum(len(v) for v in images.values())} images total")
        return images
    
    def run_comprehensive_tests(self, num_people=20):
        """
        Run comprehensive testing:
        - Genuine tests: Same person, different samples
        - Impostor tests: Different people
        
        Args:
            num_people: Number of people to test (use subset for speed)
        """
        images = self.load_all_images()
        people_ids = sorted(list(images.keys()))[:num_people]
        
        print(f"\n{'='*60}")
        print(f"Testing on {num_people} people")
        print(f"{'='*60}\n")
        
        # === GENUINE TESTS (Same Person) ===
        print("Running GENUINE tests (same person, different samples)...")
        genuine_count = 0
        
        for person_id in tqdm(people_ids, desc="Genuine tests"):
            samples = images[person_id]
            sample_ids = sorted(samples.keys())
            
            # Test all pairs of different samples for same person
            for i, sample1_id in enumerate(sample_ids):
                for sample2_id in sample_ids[i+1:]:
                    des1 = samples[sample1_id]['descriptors']
                    des2 = samples[sample2_id]['descriptors']
                    
                    num_matches = self.sift_match_safe(des1, des2)
                    
                    authenticated = num_matches >= self.match_threshold
                    
                    self.results['genuine_tests'].append({
                        'person_id': person_id,
                        'sample1': sample1_id,
                        'sample2': sample2_id,
                        'matches': num_matches,
                        'authenticated': authenticated,
                        'file1': samples[sample1_id]['filename'],
                        'file2': samples[sample2_id]['filename']
                    })
                    genuine_count += 1
        
        print(f"✓ Completed {genuine_count} genuine tests")
        
        # === IMPOSTOR TESTS (Different People) ===
        print("\nRunning IMPOSTOR tests (different people)...")
        impostor_count = 0
        
        # Test first person against all others
        for i, person1_id in enumerate(tqdm(people_ids[:10], desc="Impostor tests")):
            # Get first sample of person1
            sample1_id = sorted(images[person1_id].keys())[0]
            des1 = images[person1_id][sample1_id]['descriptors']
            
            # Test against different people
            for person2_id in people_ids[i+1:]:
                # Get first sample of person2
                sample2_id = sorted(images[person2_id].keys())[0]
                des2 = images[person2_id][sample2_id]['descriptors']
                
                num_matches = self.sift_match_safe(des1, des2)
                
                authenticated = num_matches >= self.match_threshold
                
                self.results['impostor_tests'].append({
                    'person1_id': person1_id,
                    'person2_id': person2_id,
                    'sample1': sample1_id,
                    'sample2': sample2_id,
                    'matches': num_matches,
                    'authenticated': authenticated,
                    'file1': images[person1_id][sample1_id]['filename'],
                    'file2': images[person2_id][sample2_id]['filename']
                })
                impostor_count += 1
        
        print(f"✓ Completed {impostor_count} impostor tests")
        print(f"\n{'='*60}\n")
    
    def calculate_metrics(self):
        """Calculate GAR, FAR, FRR, Accuracy"""
        genuine = self.results['genuine_tests']
        impostor = self.results['impostor_tests']
        
        # Genuine Acceptance Rate (GAR) - % of genuine users accepted
        genuine_accepted = sum(1 for t in genuine if t['authenticated'])
        GAR = (genuine_accepted / len(genuine) * 100) if genuine else 0
        
        # False Rejection Rate (FRR) - % of genuine users rejected
        FRR = 100 - GAR
        
        # False Acceptance Rate (FAR) - % of impostors accepted
        impostor_accepted = sum(1 for t in impostor if t['authenticated'])
        FAR = (impostor_accepted / len(impostor) * 100) if impostor else 0
        
        # True Rejection Rate
        TRR = 100 - FAR
        
        # Overall Accuracy
        total_correct = genuine_accepted + (len(impostor) - impostor_accepted)
        total_tests = len(genuine) + len(impostor)
        accuracy = (total_correct / total_tests * 100) if total_tests else 0
        
        metrics = {
            'GAR': GAR,
            'FRR': FRR,
            'FAR': FAR,
            'TRR': TRR,
            'Accuracy': accuracy,
            'total_genuine_tests': len(genuine),
            'genuine_accepted': genuine_accepted,
            'genuine_rejected': len(genuine) - genuine_accepted,
            'total_impostor_tests': len(impostor),
            'impostor_accepted': impostor_accepted,
            'impostor_rejected': len(impostor) - impostor_accepted
        }
        
        return metrics
    
    def generate_report(self, output_dir="Test_Results"):
        """Generate comprehensive test report"""
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = self.calculate_metrics()
        
        # Print summary
        print("\n" + "="*60)
        print("SIFT AUTHENTICATION SYSTEM - TEST RESULTS")
        print("="*60)
        print(f"\nTest Configuration:")
        print(f"  - Ratio Threshold (Lowe's): {self.ratio_threshold}")
        print(f"  - Match Threshold: {self.match_threshold} matches")
        print(f"\nTest Statistics:")
        print(f"  - Genuine Tests: {metrics['total_genuine_tests']}")
        print(f"  - Impostor Tests: {metrics['total_impostor_tests']}")
        print(f"  - Total Tests: {metrics['total_genuine_tests'] + metrics['total_impostor_tests']}")
        print(f"\nPerformance Metrics:")
        print(f"  - GAR (Genuine Acceptance Rate): {metrics['GAR']:.2f}%")
        print(f"  - FRR (False Rejection Rate): {metrics['FRR']:.2f}%")
        print(f"  - FAR (False Acceptance Rate): {metrics['FAR']:.2f}%")
        print(f"  - TRR (True Rejection Rate): {metrics['TRR']:.2f}%")
        print(f"  - Overall Accuracy: {metrics['Accuracy']:.2f}%")
        print(f"\nDetailed Results:")
        print(f"  Genuine: {metrics['genuine_accepted']} accepted, {metrics['genuine_rejected']} rejected")
        print(f"  Impostor: {metrics['impostor_rejected']} rejected, {metrics['impostor_accepted']} accepted")
        print("="*60 + "\n")
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump({
                'metrics': metrics,
                'configuration': {
                    'ratio_threshold': self.ratio_threshold,
                    'match_threshold': self.match_threshold
                },
                'timestamp': self.results['timestamp']
            }, f, indent=4)
        
        print(f"✓ Saved metrics to: {output_dir}/test_metrics.json")
        
        # Generate visualizations
        self.generate_visualizations(metrics, output_dir)
        
        # Save detailed results
        with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"✓ Saved detailed results to: {output_dir}/detailed_results.json")
        
        return metrics
    
    def generate_visualizations(self, metrics, output_dir):
        """Generate charts and visualizations for report"""
        
        # 1. Performance Metrics Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = ['GAR', 'FRR', 'FAR', 'TRR', 'Accuracy']
        values = [metrics[m] for m in metrics_to_plot]
        colors = ['green', 'orange', 'red', 'blue', 'purple']
        
        bars = ax.bar(metrics_to_plot, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('SIFT Authentication System - Performance Metrics', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved chart: {output_dir}/performance_metrics.png")
        
        # 2. Confusion Matrix Style Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = [
            [metrics['genuine_accepted'], metrics['genuine_rejected']],
            [metrics['impostor_accepted'], metrics['impostor_rejected']]
        ]
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Accepted', 'Rejected'], fontsize=12)
        ax.set_yticklabels(['Genuine', 'Impostor'], fontsize=12)
        ax.set_xlabel('System Decision', fontsize=13, fontweight='bold')
        ax.set_ylabel('Actual Identity', fontsize=13, fontweight='bold')
        ax.set_title('SIFT Authentication Results Matrix', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{data[i][j]}',
                             ha="center", va="center", color="black", 
                             fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Tests', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved chart: {output_dir}/results_matrix.png")
        
        # 3. Match Score Distribution
        genuine_matches = [t['matches'] for t in self.results['genuine_tests']]
        impostor_matches = [t['matches'] for t in self.results['impostor_tests']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histograms
        ax.hist(genuine_matches, bins=30, alpha=0.6, color='green', 
               label=f'Genuine (n={len(genuine_matches)})', edgecolor='black')
        ax.hist(impostor_matches, bins=30, alpha=0.6, color='red', 
               label=f'Impostor (n={len(impostor_matches)})', edgecolor='black')
        
        # Add threshold line
        ax.axvline(x=self.match_threshold, color='blue', linestyle='--', 
                  linewidth=2, label=f'Threshold ({self.match_threshold})')
        
        ax.set_xlabel('Number of SIFT Matches', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of SIFT Match Scores', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'match_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved chart: {output_dir}/match_distribution.png")


def main():
    """Main testing function"""
    
    print("\n" + "="*60)
    print("SIFT PALMPRINT AUTHENTICATION - DATASET TESTING")
    print("="*60 + "\n")
    
    # Configuration
    testing_dir = r"C:\Users\joelk\OneDrive\Desktop\palmprint ruofei\Palmprint_Recognition-master\Palmprint_Recognition-master\Palmprint\Palmprint\testing"
    
    # Initialize tester with your system's parameters
    tester = DatasetTester(
        testing_dir=testing_dir,
        ratio_threshold=0.75,      # Adjusted ratio threshold
        match_threshold=3          # Adjusted match threshold based on dataset characteristics
    )
    
    # Run tests (use subset for speed - increase for full testing)
    tester.run_comprehensive_tests(num_people=50)  # Test 50 people for better statistics
    
    # Generate report
    metrics = tester.generate_report(output_dir="Test_Results")
    
    print("\n" + "="*60)
    print("✓ TESTING COMPLETE!")
    print("="*60)
    print("\nGenerated files in Test_Results/:")
    print("  - test_metrics.json (metrics summary)")
    print("  - detailed_results.json (all test data)")
    print("  - performance_metrics.png (bar chart)")
    print("  - results_matrix.png (confusion matrix)")
    print("  - match_distribution.png (score distribution)")
    print("\nUse these results in your Chapter 6 report!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
