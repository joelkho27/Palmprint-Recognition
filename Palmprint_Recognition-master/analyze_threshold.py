import json
import numpy as np

# Load detailed results
with open('Test_Results/detailed_results.json', 'r') as f:
    data = json.load(f)

# Analyze match distributions
genuine_matches = [test['matches'] for test in data['genuine_tests']]
impostor_matches = [test['matches'] for test in data['impostor_tests']]

print('=== MATCH SCORE DISTRIBUTION ===')
print(f'\nGenuine Tests (Same Person):')
print(f'  Min: {min(genuine_matches)}')
print(f'  Max: {max(genuine_matches)}')
print(f'  Mean: {np.mean(genuine_matches):.2f}')
print(f'  Median: {np.median(genuine_matches):.2f}')

print(f'\nImpostor Tests (Different People):')
print(f'  Min: {min(impostor_matches)}')
print(f'  Max: {max(impostor_matches)}')
print(f'  Mean: {np.mean(impostor_matches):.2f}')
print(f'  Median: {np.median(impostor_matches):.2f}')

# Find optimal threshold
print(f'\n=== THRESHOLD ANALYSIS ===')
for threshold in [10, 15, 20, 25, 30]:
    gen_accepted = sum(1 for m in genuine_matches if m >= threshold)
    imp_accepted = sum(1 for m in impostor_matches if m >= threshold)
    
    GAR = 100 * gen_accepted / len(genuine_matches)
    FAR = 100 * imp_accepted / len(impostor_matches)
    FRR = 100 - GAR
    TRR = 100 - FAR
    
    accuracy = 100 * (gen_accepted + (len(impostor_matches) - imp_accepted)) / (len(genuine_matches) + len(impostor_matches))
    
    print(f'\nThreshold = {threshold}:')
    print(f'  GAR: {GAR:.2f}% | FRR: {FRR:.2f}%')
    print(f'  FAR: {FAR:.2f}% | TRR: {TRR:.2f}%')
    print(f'  Accuracy: {accuracy:.2f}%')
