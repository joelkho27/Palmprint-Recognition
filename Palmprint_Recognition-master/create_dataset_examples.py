"""
Create visual examples from dataset testing for report
Shows genuine match examples and impostor rejection examples
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Rectangle

def create_dataset_example_visuals():
    """Create report-quality examples from dataset testing"""
    
    # Load detailed results
    with open('Test_Results/detailed_results.json', 'r') as f:
        data = json.load(f)
    
    # Load metrics for configuration
    with open('Test_Results/test_metrics.json', 'r') as f:
        metrics_data = json.load(f)
    
    genuine_tests = data['genuine_tests']
    impostor_tests = data['impostor_tests']
    
    # Find interesting examples
    # Best genuine match
    best_genuine = max(genuine_tests, key=lambda x: x['matches'])
    
    # Worst genuine miss (should accept but rejected)
    worst_genuine = min([t for t in genuine_tests if not t['authenticated']], 
                       key=lambda x: x['matches'])
    
    # Most confident impostor rejection
    best_impostor_reject = min(impostor_tests, key=lambda x: x['matches'])
    
    # Dangerous impostor (high matches but still rejected - or accepted by mistake)
    dangerous_impostor = max(impostor_tests, key=lambda x: x['matches'])
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('SIFT System - Dataset Testing Examples', fontsize=16, fontweight='bold')
    
    examples = [
        (best_genuine, axes[0, 0], "Best Genuine Match", "green"),
        (worst_genuine, axes[0, 1], "Genuine False Reject", "orange"),
        (best_impostor_reject, axes[1, 0], "Perfect Impostor Rejection", "green"),
        (dangerous_impostor, axes[1, 1], "Close Call Impostor", "red" if dangerous_impostor['authenticated'] else "orange")
    ]
    
    for test_data, ax, title, color in examples:
        # Load images
        is_genuine = 'person_id' in test_data
        
        if is_genuine:
            img1_path = f"Palmprint/Palmprint/testing/{test_data['file1']}"
            img2_path = f"Palmprint/Palmprint/testing/{test_data['file2']}"
            person_id = test_data['person_id']
            label_text = f"Person {person_id}\nSamples: {test_data['sample1']} vs {test_data['sample2']}"
        else:
            img1_path = f"Palmprint/Palmprint/testing/{test_data['file1']}"
            img2_path = f"Palmprint/Palmprint/testing/{test_data['file2']}"
            person1 = test_data['person1_id']
            person2 = test_data['person2_id']
            label_text = f"Person {person1} vs Person {person2}"
        
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        # Create side-by-side comparison
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        h = max(h1, h2)
        combined = np.ones((h, w1 + w2 + 20), dtype=np.uint8) * 255
        combined[:h1, :w1] = img1
        combined[:h2, w1+20:] = img2
        
        # Display
        ax.imshow(combined, cmap='gray')
        ax.axis('off')
        
        # Add title
        ax.text(combined.shape[1]/2, -20, title, 
                ha='center', fontsize=12, fontweight='bold')
        
        # Add result box
        matches = test_data['matches']
        auth = test_data['authenticated']
        result_text = f"✓ AUTHENTICATED" if auth else f"✗ REJECTED"
        
        textbox = f"{label_text}\n{matches} matches\n{result_text}"
        
        ax.text(combined.shape[1]/2, combined.shape[0] + 40,
                textbox,
                ha='center', va='top',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=color, 
                         alpha=0.3, 
                         edgecolor=color, 
                         linewidth=2))
    
    plt.tight_layout()
    plt.savefig('Test_Results/dataset_examples.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Test_Results/dataset_examples.png")
    
    # Create match score comparison chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    genuine_matches = [t['matches'] for t in genuine_tests]
    impostor_matches = [t['matches'] for t in impostor_tests]
    
    bins = np.arange(0, max(max(genuine_matches), max(impostor_matches)) + 2, 1)
    
    ax.hist(genuine_matches, bins=bins, alpha=0.7, color='green', 
            label=f'Genuine (n={len(genuine_matches)})', edgecolor='black')
    ax.hist(impostor_matches, bins=bins, alpha=0.7, color='red', 
            label=f'Impostor (n={len(impostor_matches)})', edgecolor='black')
    
    # Add threshold line
    threshold = metrics_data['configuration']['match_threshold']
    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')
    
    ax.set_xlabel('Number of SIFT Matches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('SIFT Match Score Distribution: Genuine vs Impostor', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Test_Results/score_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Test_Results/score_comparison.png")
    
    # Create metrics summary card
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis('off')
    
    metrics = metrics_data['metrics']
    config = metrics_data['configuration']
    
    summary_text = f"""
    SIFT PALMPRINT AUTHENTICATION
    Dataset Testing Results
    
    ═══════════════════════════════════════
    
    TEST CONFIGURATION
    • Dataset: Tongji Palmprint Database
    • People Tested: 50
    • Total Tests: {metrics['total_genuine_tests'] + metrics['total_impostor_tests']}
    • Genuine Tests: {metrics['total_genuine_tests']}
    • Impostor Tests: {metrics['total_impostor_tests']}
    
    SYSTEM PARAMETERS
    • Lowe's Ratio: {config['ratio_threshold']}
    • Match Threshold: {config['match_threshold']} matches
    
    ═══════════════════════════════════════
    
    PERFORMANCE METRICS
    
    Overall Accuracy        {metrics['Accuracy']:.2f}%
    
    Security Metrics:
    • FAR (False Accept)    {metrics['FAR']:.2f}%  ← Excellent!
    • TRR (True Reject)     {metrics['TRR']:.2f}%
    
    User Experience:
    • GAR (Genuine Accept)  {metrics['GAR']:.2f}%
    • FRR (False Reject)    {metrics['FRR']:.2f}%
    
    ═══════════════════════════════════════
    
    DETAILED RESULTS
    
    Genuine Tests (Same Person):
    ✓ Accepted:  {metrics['genuine_accepted']} / {metrics['total_genuine_tests']}
    ✗ Rejected:  {metrics['genuine_rejected']} / {metrics['total_genuine_tests']}
    
    Impostor Tests (Different People):
    ✓ Rejected:  {metrics['impostor_rejected']} / {metrics['total_impostor_tests']}
    ✗ Accepted:  {metrics['impostor_accepted']} / {metrics['total_impostor_tests']}
    
    ═══════════════════════════════════════
    
    KEY FINDINGS
    
    ✅ High Security: Only {metrics['FAR']:.2f}% false acceptance
    ✅ Strong Accuracy: {metrics['Accuracy']:.2f}% overall correct
    ✅ Low Impostor Risk: {metrics['impostor_rejected']}/{metrics['total_impostor_tests']} rejected
    
    ⚠️  Moderate GAR: Trade-off for security
    
    """
    
    ax.text(0.5, 0.5, summary_text,
            ha='center', va='center',
            fontsize=11,
            family='monospace',
            bbox=dict(boxstyle='round,pad=1', 
                     facecolor='lightblue', 
                     alpha=0.8,
                     edgecolor='navy',
                     linewidth=3))
    
    plt.savefig('Test_Results/metrics_summary_card.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Test_Results/metrics_summary_card.png")
    
    print("\n" + "="*60)
    print("✓ All dataset example visuals created successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. dataset_examples.png       - 4 example test cases")
    print("  2. score_comparison.png       - Match score distribution")
    print("  3. metrics_summary_card.png   - Complete metrics summary")
    print("\nUse these in your Chapter 6 report!")
    print("="*60)

if __name__ == "__main__":
    create_dataset_example_visuals()
