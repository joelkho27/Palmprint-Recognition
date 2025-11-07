# SIFT Implementation Code Snippets
# Labeled and ready for report integration

---

## Code Snippet 1: SIFT Feature Extraction Configuration

```python
# Enhanced SIFT feature extractor with optimized parameters
sift = cv2.SIFT_create(
    nfeatures=1200,           # Extract up to 1200 keypoints
    contrastThreshold=0.02,   # Lower threshold detects more features
    edgeThreshold=12,         # Edge detection sensitivity
    sigma=1.6                 # Gaussian blur for scale-space
)

# Detect keypoints and compute 128-D descriptors
keypoints1, descriptors1 = sift.detectAndCompute(palm_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(palm_image2, None)
```

**Label**: Figure X.X - SIFT feature extraction with optimized parameters for palmprint recognition

---

## Code Snippet 2: Lowe's Ratio Test for Feature Matching

```python
# Brute-Force matcher for descriptor comparison
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter reliable matches
good_matches = []
ratio_threshold = 0.75  # Ratio test threshold

for match_pair in matches:
    if len(match_pair) == 2:
        best_match, second_match = match_pair
        # Accept only if best match is significantly better
        if best_match.distance < ratio_threshold * second_match.distance:
            good_matches.append(best_match)

num_matches = len(good_matches)
```

**Label**: Figure X.X - Lowe's ratio test implementation for robust feature matching

---

## Code Snippet 3: Image Preprocessing Pipeline

```python
def preprocess_palm_image(image):
    """Three-stage preprocessing for enhanced feature extraction"""
    
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Stage 1: Bilateral filtering for noise reduction with edge preservation
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Stage 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Stage 3: Adaptive gamma correction for lighting normalization
    mean_intensity = np.mean(enhanced)
    if mean_intensity < 100:
        gamma = 1.2      # Brighten dark images
    elif mean_intensity > 150:
        gamma = 0.8      # Darken bright images
    else:
        gamma = 1.0      # No adjustment needed
    
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 
                                for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lookup_table)
    
    return enhanced
```

**Label**: Figure X.X - Multi-stage preprocessing pipeline for palmprint image enhancement

---

## Code Snippet 4: Six-Layer Validation System

```python
def six_layer_authentication(image1, image2):
    """Multi-layer validation for robust authentication"""
    
    # Preprocessing
    palm1 = preprocess_palm_image(image1)
    palm2 = preprocess_palm_image(image2)
    
    # Extract SIFT features
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
    kp1, des1 = sift.detectAndCompute(palm1, None)
    kp2, des2 = sift.detectAndCompute(palm2, None)
    
    # LAYER 1: SIFT Feature Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    sift_score = len(good_matches)
    
    # LAYER 2: Texture Correlation
    texture1 = calculate_texture_histogram(palm1)
    texture2 = calculate_texture_histogram(palm2)
    texture_score = np.corrcoef(texture1, texture2)[0, 1]
    
    # LAYER 3: Geometric Similarity
    geo1 = calculate_geometric_features(kp1)
    geo2 = calculate_geometric_features(kp2)
    geometric_score = np.corrcoef(geo1, geo2)[0, 1]
    
    # LAYER 4: Template Matching
    result = cv2.matchTemplate(palm1, palm2, cv2.TM_CCOEFF_NORMED)
    _, template_score, _, _ = cv2.minMaxLoc(result)
    
    # LAYER 5: SSIM (Structural Similarity Index)
    ssim_score = calculate_ssim(palm1, palm2)
    
    # LAYER 6: Edge Similarity
    edges1 = cv2.Canny(palm1, 50, 150)
    edges2 = cv2.Canny(palm2, 50, 150)
    edge_score = np.sum(edges1 & edges2) / (np.sum(edges1 | edges2) + 1e-7)
    
    # Validation: Check all thresholds
    tests_passed = (
        sift_score >= 60 and
        texture_score >= 0.75 and
        geometric_score >= 0.70 and
        template_score >= 0.75 and
        ssim_score >= 0.55 and
        edge_score >= 0.70
    )
    
    # Composite score calculation
    final_score = (sift_score * 3.0 + texture_score * 120 + 
                  geometric_score * 80 + template_score * 100 + 
                  ssim_score * 150 + edge_score * 80)
    
    # Authentication decision
    authenticated = tests_passed and final_score >= 300
    
    return authenticated, final_score
```

**Label**: Figure X.X - Six-layer validation architecture for palmprint authentication

---

## Code Snippet 5: Texture Histogram Calculation

```python
def calculate_texture_histogram(image):
    """Extract texture features using gradient magnitude histogram"""
    
    # Calculate gradients in X and Y directions using Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Create normalized histogram with 64 bins
    histogram, _ = np.histogram(magnitude.ravel(), bins=64, range=(0, 255))
    histogram = histogram.astype(float)
    histogram = histogram / (histogram.sum() + 1e-7)  # Normalize
    
    return histogram
```

**Label**: Figure X.X - Texture feature extraction using gradient magnitude histograms

---

## Code Snippet 6: Geometric Feature Extraction

```python
def calculate_geometric_features(keypoints):
    """Extract geometric properties from keypoint distribution"""
    
    if len(keypoints) < 5:
        return np.zeros(6)
    
    # Extract keypoint coordinates
    points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    # Calculate center of keypoint cloud
    center = np.mean(points, axis=0)
    
    # Calculate distances from center
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    
    # Calculate angular distribution from center
    angles = np.arctan2(points[:, 1] - center[1], 
                       points[:, 0] - center[0])
    
    # Extract 6-dimensional geometric feature vector
    features = np.array([
        np.mean(distances),                      # Average radial distance
        np.std(distances),                       # Radial spread
        len(keypoints),                          # Keypoint count
        np.std(angles),                          # Angular distribution
        np.max(distances) - np.min(distances),   # Radial range
        np.mean([kp.response for kp in keypoints])  # Average response
    ])
    
    return features
```

**Label**: Figure X.X - Geometric feature calculation from SIFT keypoint spatial distribution

---

## Code Snippet 7: SSIM Calculation

```python
def calculate_ssim(image1, image2):
    """Calculate Structural Similarity Index (SSIM)"""
    
    # Resize images for consistent comparison
    img1 = cv2.resize(image1, (256, 256)).astype(np.float64)
    img2 = cv2.resize(image2, (256, 256)).astype(np.float64)
    
    # Calculate local means using Gaussian window
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    # Calculate squared means and cross-product
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM constants for stability
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    
    return np.mean(ssim_map)
```

**Label**: Figure X.X - SSIM (Structural Similarity Index) implementation for perceptual similarity

---

## Code Snippet 8: Dataset Testing - Genuine Tests

```python
def run_genuine_tests(images, num_people=50):
    """Test same person with different samples (intra-class matching)"""
    
    genuine_results = []
    people_ids = sorted(list(images.keys()))[:num_people]
    
    for person_id in people_ids:
        samples = images[person_id]
        sample_ids = sorted(samples.keys())
        
        # Test all pairwise combinations of different samples
        for i, sample1_id in enumerate(sample_ids):
            for sample2_id in sample_ids[i+1:]:
                # Extract descriptors
                des1 = samples[sample1_id]['descriptors']
                des2 = samples[sample2_id]['descriptors']
                
                # Match features
                num_matches = sift_match_safe(des1, des2)
                
                # Check if authenticated
                authenticated = num_matches >= match_threshold
                
                genuine_results.append({
                    'person_id': person_id,
                    'matches': num_matches,
                    'authenticated': authenticated
                })
    
    return genuine_results
```

**Label**: Figure X.X - Genuine test implementation for intra-class matching evaluation

---

## Code Snippet 9: Dataset Testing - Impostor Tests

```python
def run_impostor_tests(images, num_people=50):
    """Test different people (inter-class matching)"""
    
    impostor_results = []
    people_ids = sorted(list(images.keys()))[:num_people]
    
    # Test each person against all other people
    for i, person1_id in enumerate(people_ids):
        # Get first sample of person1
        sample1_id = sorted(images[person1_id].keys())[0]
        des1 = images[person1_id][sample1_id]['descriptors']
        
        # Test against different people
        for person2_id in people_ids[i+1:]:
            # Get first sample of person2
            sample2_id = sorted(images[person2_id].keys())[0]
            des2 = images[person2_id][sample2_id]['descriptors']
            
            # Match features
            num_matches = sift_match_safe(des1, des2)
            
            # Should NOT authenticate (different people)
            authenticated = num_matches >= match_threshold
            
            impostor_results.append({
                'person1_id': person1_id,
                'person2_id': person2_id,
                'matches': num_matches,
                'authenticated': authenticated
            })
    
    return impostor_results
```

**Label**: Figure X.X - Impostor test implementation for inter-class rejection evaluation

---

## Code Snippet 10: Performance Metrics Calculation

```python
def calculate_biometric_metrics(genuine_results, impostor_results):
    """Calculate GAR, FAR, FRR, TRR, and Accuracy"""
    
    # Count genuine acceptances and rejections
    genuine_accepted = sum(1 for t in genuine_results if t['authenticated'])
    genuine_rejected = len(genuine_results) - genuine_accepted
    
    # Count impostor acceptances and rejections
    impostor_accepted = sum(1 for t in impostor_results if t['authenticated'])
    impostor_rejected = len(impostor_results) - impostor_accepted
    
    # Calculate rates
    GAR = (genuine_accepted / len(genuine_results)) * 100  # Genuine Acceptance Rate
    FRR = (genuine_rejected / len(genuine_results)) * 100  # False Rejection Rate
    FAR = (impostor_accepted / len(impostor_results)) * 100  # False Acceptance Rate
    TRR = (impostor_rejected / len(impostor_results)) * 100  # True Rejection Rate
    
    # Overall accuracy
    total_correct = genuine_accepted + impostor_rejected
    total_tests = len(genuine_results) + len(impostor_results)
    accuracy = (total_correct / total_tests) * 100
    
    metrics = {
        'GAR': GAR,      # Higher is better (user convenience)
        'FAR': FAR,      # Lower is better (security)
        'FRR': FRR,      # Lower is better (user convenience)
        'TRR': TRR,      # Higher is better (security)
        'Accuracy': accuracy
    }
    
    return metrics
```

**Label**: Figure X.X - Biometric performance metrics calculation following ISO/IEC 19795 standards

---

## Code Snippet 11: Safe SIFT Matching with Error Handling

```python
def sift_match_safe(descriptors1, descriptors2, ratio_threshold=0.75):
    """
    Safe SIFT matching with error handling for production use
    Prevents crashes when insufficient features are detected
    """
    
    # Validate descriptors exist and have minimum features
    if descriptors1 is None or descriptors2 is None:
        return 0
    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return 0
    
    try:
        # Brute-force matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        match_count = 0
        
        # Safe iteration with validation
        for match_pair in matches:
            if len(match_pair) == 2:  # Ensure pair has both matches
                first, second = match_pair
                # Apply Lowe's ratio test
                if first.distance < ratio_threshold * second.distance:
                    match_count += 1
        
        return match_count
        
    except Exception as e:
        # Return 0 on any error to prevent crashes
        return 0
```

**Label**: Figure X.X - Production-ready SIFT matching with comprehensive error handling

---

## Code Snippet 12: Authentication Threshold Configuration

```python
# Authentication thresholds for six-layer validation
AUTHENTICATION_THRESHOLDS = {
    # Layer 1: SIFT Feature Matching
    'min_sift_matches': 60,          # Minimum SIFT matches required
    
    # Layer 2: Texture Correlation
    'min_texture_correlation': 0.75,  # Texture similarity threshold
    
    # Layer 3: Geometric Similarity
    'min_geometric_similarity': 0.70, # Geometric consistency threshold
    
    # Layer 4: Template Matching
    'min_template_score': 0.75,       # Template correlation threshold
    
    # Layer 5: SSIM
    'min_ssim_score': 0.55,           # Structural similarity threshold
    
    # Layer 6: Edge Similarity
    'min_edge_similarity': 0.70,      # Edge correlation threshold
    
    # Global Decision Criteria
    'required_passed_tests': 6,       # ALL 6 tests must pass
    'final_score_threshold': 300      # Minimum composite score
}

# Dataset testing thresholds (adjusted for lower quality images)
DATASET_THRESHOLDS = {
    'ratio_threshold': 0.75,          # Lowe's ratio test
    'match_threshold': 3              # Minimum matches for dataset
}
```

**Label**: Table X.X - Authentication threshold configuration for SIFT-based system

---

## Quick Reference Table

| Snippet # | Title | Use In Section |
|-----------|-------|----------------|
| 1 | SIFT Configuration | Methods - Feature Extraction |
| 2 | Lowe's Ratio Test | Methods - Feature Matching |
| 3 | Preprocessing Pipeline | Methods - Image Preprocessing |
| 4 | Six-Layer Validation | Methods - Authentication System |
| 5 | Texture Histogram | Implementation - Layer 2 |
| 6 | Geometric Features | Implementation - Layer 3 |
| 7 | SSIM Calculation | Implementation - Layer 5 |
| 8 | Genuine Tests | Testing - Methodology |
| 9 | Impostor Tests | Testing - Methodology |
| 10 | Metrics Calculation | Testing - Performance Evaluation |
| 11 | Safe Matching | Implementation - Error Handling |
| 12 | Threshold Config | Configuration - Parameters |

---

**Total Code Snippets**: 12
**Lines of Code**: ~450 lines
**Coverage**: Complete SIFT implementation from preprocessing to evaluation
