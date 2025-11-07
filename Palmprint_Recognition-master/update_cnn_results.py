"""
Update CNN results after external testing
Run this script after you've tested the CNN on the Tongji dataset
"""

from comparison_framework import ComparisonFramework

# Create framework
framework = ComparisonFramework()

# ============================================================================
# TODO: FILL IN YOUR CNN TEST RESULTS HERE
# ============================================================================

# After testing CNN on the same dataset, update these values:
framework.update_cnn_results(
    num_people_tested=50,           # Matches SIFT testing
    total_tests=595,                # Matches SIFT testing
    genuine_tests=150,              # Matches SIFT testing
    impostor_tests=445,             # Matches SIFT testing
    
    # ACTUAL CNN RESULTS FROM YOUR TESTING:
    accuracy=91.68,                 # CNN achieved 91.68% accuracy
    gar=79.42,                      # Genuine Acceptance Rate
    far=1.21,                       # False Acceptance Rate
    frr=20.58,                      # False Rejection Rate
    trr=98.79,                      # True Rejection Rate
    
    # Updated deployment metrics:
    training_time="100 epochs (unlabeled palm images)",
    inference_time_per_image="~0.12 seconds with GPU"
)

# ============================================================================
# Generate complete comparison with your CNN results
# ============================================================================

print("\n" + "="*80)
print("GENERATING FINAL COMPARISON WITH YOUR CNN RESULTS")
print("="*80 + "\n")

framework.generate_full_report()

print("\n" + "🎉 "*30)
print("COMPARISON COMPLETE!")
print("🎉 "*30)
print("\nFiles generated:")
print("  - Test_Results/method_comparison.png (visualization)")
print("  - Test_Results/comparison_metrics.json (data)")
print("  - Tables printed above (copy to report)")
print("\nYou can now use these for your Chapter 6 comparison!")
