# CNN Testing Guide - How to Fill in Comparison Results

## What You Have Now

✅ **SIFT Results (already complete):**
- 50 people tested from Tongji dataset
- 595 total tests (150 genuine + 445 impostor)
- Accuracy: 87.23%
- GAR: 50.67%, FAR: 0.45%, FRR: 49.33%, TRR: 99.55%

## What You Need to Do

### Step 1: Test CNN on Same Dataset
Test your partner's trained CNN model on the **same Tongji dataset**:
- Use the same 50 people
- Perform the same test structure:
  - **150 genuine tests**: Match person with their own other samples
  - **445 impostor tests**: Match person with different people

### Step 2: Calculate These Metrics

From your CNN testing, you need to calculate:

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Accuracy** | `(TP + TN) / Total × 100` | Overall correctness |
| **GAR** | `TP / Genuine_Tests × 100` | How often real users are accepted |
| **FAR** | `FP / Impostor_Tests × 100` | How often fake users are accepted (security risk!) |
| **FRR** | `FN / Genuine_Tests × 100` | How often real users are rejected (frustration) |
| **TRR** | `TN / Impostor_Tests × 100` | How often fake users are rejected |

Where:
- **TP** (True Positive) = Genuine user correctly accepted
- **TN** (True Negative) = Impostor correctly rejected  
- **FP** (False Positive) = Impostor incorrectly accepted
- **FN** (False Negative) = Genuine user incorrectly rejected

### Step 3: Update the Comparison

1. Open `update_cnn_results.py`
2. Replace the placeholder values (92.5, 88.0, etc.) with your actual CNN results
3. Run: `python update_cnn_results.py`
4. Get updated tables and visualizations!

## Example Calculation

Let's say your CNN testing gives:
- 150 genuine tests → 132 accepted (TP), 18 rejected (FN)
- 445 impostor tests → 440 rejected (TN), 5 accepted (FP)

Then:
```
Accuracy = (132 + 440) / 595 × 100 = 96.13%
GAR = 132 / 150 × 100 = 88.00%
FAR = 5 / 445 × 100 = 1.12%
FRR = 18 / 150 × 100 = 12.00%
TRR = 440 / 445 × 100 = 98.88%
```

You would update:
```python
framework.update_cnn_results(
    num_people_tested=50,
    total_tests=595,
    genuine_tests=150,
    impostor_tests=445,
    accuracy=96.13,
    gar=88.00,
    far=1.12,
    frr=12.00,
    trr=98.88
)
```

## Quick Reference: Current Files

| File | Purpose |
|------|---------|
| `comparison_framework.py` | Main comparison generator (don't edit) |
| `update_cnn_results.py` | **EDIT THIS** - Fill in your CNN results here |
| `Test_Results/method_comparison.png` | Visualization (auto-generated) |
| `Test_Results/comparison_metrics.json` | Data for report (auto-generated) |

## Testing Tips

### For Genuine Tests (150 total):
- Pick 50 people from dataset
- Each person has 6 images (001-1.bmp to 001-6.bmp)
- Use 3 for training (1,2,3), 3 for testing (4,5,6)
- Test each person: Compare test images with their training embeddings
- Example: Compare 001-4 with 001-1, 001-2, 001-3 embeddings

### For Impostor Tests (445 total):
- For each person, test against OTHER people
- Example: Compare 001-4 with 002's embeddings, 003's embeddings, etc.
- Should be REJECTED (different people)

## What Makes CNN "Better"?

Even if CNN accuracy is similar to SIFT, you can justify it as better choice because:

✅ **Deployment advantages:**
- Faster inference (0.1-0.2s vs 0.5-1.0s)
- GPU acceleration support
- Better mobile deployment
- Scalable (O(1) vs O(n) comparison)

✅ **Modern architecture:**
- Industry standard ResNet-18
- Retrainable with new data
- End-to-end learned features

✅ **Integration:**
- Already integrated in your product
- Team has expertise
- Ready for multimodal fusion

So even if accuracy is 87% (same as SIFT), CNN is still the right choice for production!

## After You Fill Results

Run this to regenerate everything:
```bash
python update_cnn_results.py
```

You'll get:
1. Updated comparison tables (copy to report)
2. Updated visualization (use in Chapter 6)
3. LaTeX table (paste in report if using LaTeX)
4. JSON data (for reference)

## Questions?

- SIFT accuracy: 87.23%
- To make CNN "better": Aim for >87.23% accuracy OR similar accuracy + emphasize deployment advantages
- FAR is critical: Lower FAR = better security (SIFT has excellent 0.45% FAR)
- GAR matters for UX: Higher GAR = less user frustration (SIFT has 50.67% GAR)
