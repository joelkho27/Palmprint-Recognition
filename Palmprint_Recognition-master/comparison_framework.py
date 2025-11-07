"""
Comparison Framework for SIFT vs CNN
Creates tables and visualizations for comparative analysis
Fill in CNN results after testing on Tongji dataset
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class ComparisonFramework:
    def __init__(self):
        # SIFT results (already tested)
        self.sift_results = {
            "method": "SIFT + 6-Layer Validation",
            "dataset": "Tongji Palmprint Database",
            "num_people_tested": 50,
            "total_tests": 595,
            "genuine_tests": 150,
            "impostor_tests": 445,
            "accuracy": 87.23,
            "gar": 50.67,  # Genuine Acceptance Rate
            "far": 0.45,   # False Acceptance Rate
            "frr": 49.33,  # False Rejection Rate
            "trr": 99.55,  # True Rejection Rate
            "eer": None,   # Equal Error Rate (can calculate if needed)
            "training_required": False,
            "training_time": "0 hours (no training needed)",
            "inference_time_per_image": "~0.5-1.0 seconds",
            "model_size": "N/A (algorithm-based)",
            "features_detected_avg": "200-1200 keypoints"
        }
        
        # CNN results placeholder - YOU FILL THESE IN AFTER TESTING
        self.cnn_results = {
            "method": "ResNet-18 Contrastive Learning",
            "dataset": "Tongji Palmprint Database",
            "num_people_tested": None,  # TODO: Fill after testing
            "total_tests": None,  # TODO: Fill after testing
            "genuine_tests": None,  # TODO: Fill after testing
            "impostor_tests": None,  # TODO: Fill after testing
            "accuracy": None,  # TODO: Fill after testing
            "gar": None,  # TODO: Fill after testing
            "far": None,  # TODO: Fill after testing
            "frr": None,  # TODO: Fill after testing
            "trr": None,  # TODO: Fill after testing
            "eer": None,  # TODO: Fill after testing (optional)
            "training_required": True,
            "training_time": "100 epochs (~X hours)",  # TODO: Fill actual time
            "inference_time_per_image": "~0.1-0.2 seconds",  # TODO: Measure actual
            "model_size": "~45 MB (ResNet-18)",
            "embedding_dimension": 256
        }
    
    def update_cnn_results(self, **kwargs):
        """
        Update CNN results after testing
        
        Example usage:
        framework.update_cnn_results(
            num_people_tested=50,
            total_tests=595,
            genuine_tests=150,
            impostor_tests=445,
            accuracy=92.5,
            gar=88.0,
            far=1.2,
            frr=12.0,
            trr=98.8
        )
        """
        for key, value in kwargs.items():
            if key in self.cnn_results:
                self.cnn_results[key] = value
    
    def generate_performance_table(self):
        """Generate performance metrics comparison table"""
        print("\n" + "="*80)
        print("PERFORMANCE METRICS COMPARISON")
        print("="*80)
        print(f"{'Metric':<30} {'SIFT':<20} {'CNN (ResNet-18)':<20}")
        print("-"*80)
        
        metrics = [
            ("Dataset", "dataset"),
            ("People Tested", "num_people_tested"),
            ("Total Tests", "total_tests"),
            ("Genuine Tests", "genuine_tests"),
            ("Impostor Tests", "impostor_tests"),
            ("Overall Accuracy (%)", "accuracy"),
            ("GAR - Genuine Accept (%)", "gar"),
            ("FAR - False Accept (%)", "far"),
            ("FRR - False Reject (%)", "frr"),
            ("TRR - True Reject (%)", "trr"),
        ]
        
        for label, key in metrics:
            sift_val = self.sift_results[key]
            cnn_val = self.cnn_results[key]
            
            if isinstance(sift_val, float):
                sift_str = f"{sift_val:.2f}%"
                cnn_str = f"{cnn_val:.2f}%" if cnn_val is not None else "TODO: Test"
            elif isinstance(sift_val, int):
                sift_str = str(sift_val)
                cnn_str = str(cnn_val) if cnn_val is not None else "TODO: Test"
            else:
                sift_str = str(sift_val)
                cnn_str = str(cnn_val) if cnn_val is not None else "TODO: Test"
            
            print(f"{label:<30} {sift_str:<20} {cnn_str:<20}")
        
        print("="*80 + "\n")
    
    def generate_deployment_table(self):
        """Generate deployment characteristics comparison"""
        print("\n" + "="*80)
        print("DEPLOYMENT CHARACTERISTICS COMPARISON")
        print("="*80)
        print(f"{'Characteristic':<30} {'SIFT':<25} {'CNN (ResNet-18)':<25}")
        print("-"*80)
        
        deployment = [
            ("Training Required", 
             "No (algorithm-based)", 
             "Yes (100 epochs)"),
            ("Training Time", 
             self.sift_results["training_time"],
             self.cnn_results["training_time"]),
            ("Inference Speed", 
             self.sift_results["inference_time_per_image"],
             self.cnn_results["inference_time_per_image"]),
            ("Model Size", 
             self.sift_results["model_size"],
             self.cnn_results["model_size"]),
            ("GPU Acceleration", 
             "Not utilized", 
             "Yes (PyTorch)"),
            ("Mobile Deployment", 
             "Moderate (CPU-heavy)", 
             "Excellent (quantizable)"),
            ("Scalability", 
             "O(n) feature matching", 
             "O(1) embedding comparison"),
            ("Adaptability", 
             "Fixed algorithm", 
             "Retrainable with new data"),
        ]
        
        for char, sift_val, cnn_val in deployment:
            print(f"{char:<30} {sift_val:<25} {cnn_val:<25}")
        
        print("="*80 + "\n")
    
    def generate_visualization(self, save_path="comparison_visualization.png"):
        """Generate comprehensive comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SIFT vs CNN Performance Comparison', fontsize=16, fontweight='bold')
        
        # Check if CNN results are filled
        cnn_tested = self.cnn_results["accuracy"] is not None
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        methods = ['SIFT', 'CNN']
        if cnn_tested:
            accuracies = [self.sift_results["accuracy"], self.cnn_results["accuracy"]]
            colors = ['#3498db', '#e74c3c']
            bars = ax1.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax1.set_ylim(0, 100)
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        else:
            ax1.text(0.5, 0.5, 'Fill CNN results\nto generate chart', 
                    ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 2. GAR vs FAR Comparison
        ax2 = axes[0, 1]
        if cnn_tested:
            x = np.arange(2)
            width = 0.35
            gar_vals = [self.sift_results["gar"], self.cnn_results["gar"]]
            far_vals = [self.sift_results["far"], self.cnn_results["far"]]
            
            bars1 = ax2.bar(x - width/2, gar_vals, width, label='GAR (higher better)', 
                           color='#2ecc71', alpha=0.7, edgecolor='black')
            bars2 = ax2.bar(x + width/2, far_vals, width, label='FAR (lower better)', 
                           color='#e67e22', alpha=0.7, edgecolor='black')
            
            ax2.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
            ax2.set_title('GAR vs FAR Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Fill CNN results\nto generate chart', 
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        
        # 3. FRR vs TRR Comparison
        ax3 = axes[1, 0]
        if cnn_tested:
            x = np.arange(2)
            frr_vals = [self.sift_results["frr"], self.cnn_results["frr"]]
            trr_vals = [self.sift_results["trr"], self.cnn_results["trr"]]
            
            bars1 = ax3.bar(x - width/2, frr_vals, width, label='FRR (lower better)', 
                           color='#e74c3c', alpha=0.7, edgecolor='black')
            bars2 = ax3.bar(x + width/2, trr_vals, width, label='TRR (higher better)', 
                           color='#3498db', alpha=0.7, edgecolor='black')
            
            ax3.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
            ax3.set_title('FRR vs TRR Comparison', fontsize=14, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(methods)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Fill CNN results\nto generate chart', 
                    ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        
        # 4. Summary Score Card
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "COMPARISON SUMMARY\n" + "="*40 + "\n\n"
        
        if cnn_tested:
            # Determine winner for each metric
            acc_winner = "CNN" if self.cnn_results["accuracy"] > self.sift_results["accuracy"] else "SIFT"
            gar_winner = "CNN" if self.cnn_results["gar"] > self.sift_results["gar"] else "SIFT"
            far_winner = "CNN" if self.cnn_results["far"] < self.sift_results["far"] else "SIFT"
            frr_winner = "CNN" if self.cnn_results["frr"] < self.sift_results["frr"] else "SIFT"
            
            summary_text += f"Accuracy Winner: {acc_winner}\n"
            summary_text += f"  SIFT: {self.sift_results['accuracy']:.2f}%\n"
            summary_text += f"  CNN:  {self.cnn_results['accuracy']:.2f}%\n\n"
            
            summary_text += f"Security (Lower FAR): {far_winner}\n"
            summary_text += f"  SIFT FAR: {self.sift_results['far']:.2f}%\n"
            summary_text += f"  CNN FAR:  {self.cnn_results['far']:.2f}%\n\n"
            
            summary_text += f"User Experience (Higher GAR): {gar_winner}\n"
            summary_text += f"  SIFT GAR: {self.sift_results['gar']:.2f}%\n"
            summary_text += f"  CNN GAR:  {self.cnn_results['gar']:.2f}%\n\n"
            
            summary_text += "Deployment Advantages:\n"
            summary_text += "  CNN: Faster inference, GPU support,\n"
            summary_text += "       mobile-ready, scalable\n"
            summary_text += "  SIFT: No training needed, interpretable,\n"
            summary_text += "        proven traditional method\n"
        else:
            summary_text += "Fill CNN results to see\n"
            summary_text += "detailed comparison summary.\n\n"
            summary_text += "Required metrics:\n"
            summary_text += "  - accuracy\n"
            summary_text += "  - gar, far, frr, trr\n"
            summary_text += "  - num_people_tested\n"
            summary_text += "  - total_tests\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()
    
    def save_comparison_json(self, save_path="comparison_metrics.json"):
        """Save comparison data as JSON for report integration"""
        comparison_data = {
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sift_results": self.sift_results,
            "cnn_results": self.cnn_results,
            "notes": "CNN results filled in after external testing on same dataset"
        }
        
        with open(save_path, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        
        print(f"Comparison data saved to: {save_path}")
    
    def generate_latex_table(self):
        """Generate LaTeX table for report"""
        print("\n" + "="*80)
        print("LATEX TABLE FOR REPORT")
        print("="*80)
        
        latex = r"""
\begin{table}[h]
\centering
\caption{Performance Comparison: SIFT vs CNN on Tongji Palmprint Database}
\label{tab:performance_comparison}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Metric} & \textbf{SIFT} & \textbf{CNN (ResNet-18)} \\
\hline
\hline
People Tested & """ + str(self.sift_results['num_people_tested']) + r""" & """ + \
        (str(self.cnn_results['num_people_tested']) if self.cnn_results['num_people_tested'] else "TODO") + r""" \\
Total Tests & """ + str(self.sift_results['total_tests']) + r""" & """ + \
        (str(self.cnn_results['total_tests']) if self.cnn_results['total_tests'] else "TODO") + r""" \\
\hline
Accuracy (\%) & """ + f"{self.sift_results['accuracy']:.2f}" + r""" & """ + \
        (f"{self.cnn_results['accuracy']:.2f}" if self.cnn_results['accuracy'] else "TODO") + r""" \\
GAR (\%) & """ + f"{self.sift_results['gar']:.2f}" + r""" & """ + \
        (f"{self.cnn_results['gar']:.2f}" if self.cnn_results['gar'] else "TODO") + r""" \\
FAR (\%) & """ + f"{self.sift_results['far']:.2f}" + r""" & """ + \
        (f"{self.cnn_results['far']:.2f}" if self.cnn_results['far'] else "TODO") + r""" \\
FRR (\%) & """ + f"{self.sift_results['frr']:.2f}" + r""" & """ + \
        (f"{self.cnn_results['frr']:.2f}" if self.cnn_results['frr'] else "TODO") + r""" \\
TRR (\%) & """ + f"{self.sift_results['trr']:.2f}" + r""" & """ + \
        (f"{self.cnn_results['trr']:.2f}" if self.cnn_results['trr'] else "TODO") + r""" \\
\hline
Training Required & No & Yes (100 epochs) \\
Inference Speed & 0.5-1.0s & 0.1-0.2s \\
Model Size & N/A & 45 MB \\
\hline
\end{tabular}
\end{table}
"""
        print(latex)
        print("="*80 + "\n")
    
    def generate_full_report(self):
        """Generate all comparison materials"""
        print("\n" + "🔬 "*30)
        print("GENERATING COMPREHENSIVE COMPARISON REPORT")
        print("🔬 "*30 + "\n")
        
        # 1. Performance table
        self.generate_performance_table()
        
        # 2. Deployment table
        self.generate_deployment_table()
        
        # 3. Visualization
        self.generate_visualization("Test_Results/method_comparison.png")
        
        # 4. Save JSON
        self.save_comparison_json("Test_Results/comparison_metrics.json")
        
        # 5. LaTeX table
        self.generate_latex_table()
        
        print("\n" + "✅ "*30)
        print("ALL COMPARISON MATERIALS GENERATED!")
        print("✅ "*30 + "\n")
        
        if self.cnn_results["accuracy"] is None:
            print("⚠️  NEXT STEP: Test CNN on Tongji dataset and fill in results\n")
            print("Use: framework.update_cnn_results(accuracy=XX, gar=XX, ...)")
        else:
            print("✅ CNN results filled in! Comparison complete.\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create framework with SIFT results already filled
    framework = ComparisonFramework()
    
    print("="*80)
    print("STEP 1: Generate comparison with placeholder CNN results")
    print("="*80)
    framework.generate_full_report()
    
    print("\n" + "="*80)
    print("STEP 2: After you test CNN, update results like this:")
    print("="*80)
    print("""
# Example: Update CNN results after testing
framework.update_cnn_results(
    num_people_tested=50,
    total_tests=595,
    genuine_tests=150,
    impostor_tests=445,
    accuracy=92.5,      # TODO: Replace with actual
    gar=88.0,           # TODO: Replace with actual
    far=1.2,            # TODO: Replace with actual
    frr=12.0,           # TODO: Replace with actual
    trr=98.8            # TODO: Replace with actual
)

# Then regenerate everything with real data
framework.generate_full_report()
""")
    
    print("\n" + "="*80)
    print("METRICS YOU NEED TO CALCULATE FROM CNN TESTING:")
    print("="*80)
    print("""
1. num_people_tested: How many people from dataset (recommend 50 to match SIFT)
2. total_tests: Total authentication attempts
3. genuine_tests: Tests with same person (different samples)
4. impostor_tests: Tests with different people
5. accuracy: (TP + TN) / total_tests * 100
6. GAR: Genuine Acceptance Rate = TP / genuine_tests * 100
7. FAR: False Acceptance Rate = FP / impostor_tests * 100
8. FRR: False Rejection Rate = FN / genuine_tests * 100
9. TRR: True Rejection Rate = TN / impostor_tests * 100

Where:
- TP (True Positive): Genuine user correctly accepted
- TN (True Negative): Impostor correctly rejected
- FP (False Positive): Impostor incorrectly accepted
- FN (False Negative): Genuine user incorrectly rejected
""")
