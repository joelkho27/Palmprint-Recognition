"""
SIFT PALMPRINT AUTHENTICATION - CODE SNIPPETS FOR REPORT
Copy and paste these code chunks directly into your report
"""

# ============================================================================
# CODE CHUNK 1: EXTRACT REGION OF INTEREST
# ============================================================================

def extract_palm_region_enhanced(img):
    """
    Extract and preprocess palm region of interest from image
    Applies noise reduction, contrast enhancement, and lighting normalization
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    height, width = gray.shape
    
    # Extract center region (palm area)
    center_x, center_y = width // 2, height // 2
    region_size = min(width, height) // 2
    
    palm_region = gray[
        max(0, center_y - region_size):min(height, center_y + region_size), 
        max(0, center_x - region_size):min(width, center_x + region_size)
    ]
    
    if palm_region.size == 0:
        return gray
    
    # Enhanced preprocessing pipeline
    try:
        # Step 1: Bilateral filtering for noise reduction (preserves edges)
        denoised = cv2.bilateralFilter(palm_region, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Step 2: CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Adaptive gamma correction for lighting normalization
        mean_intensity = np.mean(enhanced)
        if mean_intensity < 100:
            gamma = 1.2  # Brighten dark images
        elif mean_intensity > 150:
            gamma = 0.8  # Darken bright images
        else:
            gamma = 1.0  # No adjustment needed
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    except:
        return palm_region


# ============================================================================
# CODE CHUNK 2: LOAD MODEL AND EXTRACT FEATURES
# ============================================================================

def load_model_and_extract_features(palm_image):
    """
    Load SIFT detector and extract features from palm image
    Returns keypoints and 128-dimensional descriptors
    """
    # Initialize SIFT detector with optimized parameters
    sift = cv2.SIFT_create(
        nfeatures=1200,           # Maximum number of features to extract
        contrastThreshold=0.02,   # Lower threshold detects more features
        edgeThreshold=12,         # Edge detection sensitivity
        sigma=1.6                 # Gaussian blur sigma for scale-space
    )
    
    # Preprocess the palm image first
    palm_roi = extract_palm_region_enhanced(palm_image)
    
    # Detect keypoints and compute descriptors
    # keypoints: List of cv2.KeyPoint objects with location, scale, orientation
    # descriptors: Numpy array of shape (n_keypoints, 128)
    keypoints, descriptors = sift.detectAndCompute(palm_roi, None)
    
    # Validate feature extraction
    if keypoints is None or descriptors is None:
        print("⚠️  No features detected!")
        return None, None
    
    print(f"✓ Extracted {len(keypoints)} keypoints")
    print(f"✓ Descriptor shape: {descriptors.shape}")
    
    return keypoints, descriptors


# ============================================================================
# CODE CHUNK 3: COMPARE PALMS AND RETURN RESULTS
# ============================================================================

def compare_palms_and_return_results(palm_image1, palm_image2, 
                                     ratio_threshold=0.75, 
                                     match_threshold=60):
    """
    Compare two palm images using SIFT features
    Returns authentication decision and detailed matching results
    
    Args:
        palm_image1: First palm image (captured/query)
        palm_image2: Second palm image (registered/template)
        ratio_threshold: Lowe's ratio test threshold (default: 0.75)
        match_threshold: Minimum matches required for authentication (default: 60)
    
    Returns:
        is_authenticated: Boolean - True if palms match
        match_count: Integer - Number of good matches found
        results_dict: Dictionary with detailed matching information
    """
    
    # Step 1: Extract features from both palm images
    keypoints1, descriptors1 = load_model_and_extract_features(palm_image1)
    keypoints2, descriptors2 = load_model_and_extract_features(palm_image2)
    
    # Validate that features were extracted
    if descriptors1 is None or descriptors2 is None:
        return False, 0, {
            'status': 'FAILED',
            'error': 'Insufficient features detected',
            'matches': 0
        }
    
    # Step 2: Match descriptors using Brute-Force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Step 3: Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:  # Ensure we have 2 nearest neighbors
            best_match, second_match = match_pair
            # Accept match only if best is significantly better than second
            if best_match.distance < ratio_threshold * second_match.distance:
                good_matches.append(best_match)
    
    match_count = len(good_matches)
    
    # Step 4: Make authentication decision
    is_authenticated = match_count >= match_threshold
    
    # Step 5: Calculate match percentage
    max_possible_matches = min(len(keypoints1), len(keypoints2))
    match_percentage = (match_count / max_possible_matches * 100) if max_possible_matches > 0 else 0
    
    # Step 6: Prepare detailed results
    results_dict = {
        'status': 'AUTHENTICATED' if is_authenticated else 'REJECTED',
        'match_count': match_count,
        'match_threshold': match_threshold,
        'keypoints_image1': len(keypoints1),
        'keypoints_image2': len(keypoints2),
        'match_percentage': match_percentage,
        'ratio_threshold': ratio_threshold,
        'decision': 'ACCEPT' if is_authenticated else 'REJECT'
    }
    
    # Step 7: Print results
    print(f"\n{'='*60}")
    print(f"PALM COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Keypoints detected:")
    print(f"  Image 1: {len(keypoints1)} keypoints")
    print(f"  Image 2: {len(keypoints2)} keypoints")
    print(f"\nMatching results:")
    print(f"  Good matches: {match_count}")
    print(f"  Match threshold: {match_threshold}")
    print(f"  Match percentage: {match_percentage:.2f}%")
    print(f"\nDecision: {results_dict['status']}")
    print(f"{'='*60}\n")
    
    return is_authenticated, match_count, results_dict


# ============================================================================
# CODE CHUNK 4: SIX-LAYER VALIDATION SYSTEM
# ============================================================================

def six_layer_authentication(image1, image2):
    """
    Multi-layer validation for robust authentication
    All 6 layers must pass for authentication acceptance
    """
    # Preprocessing
    palm1 = extract_palm_region_enhanced(image1)
    palm2 = extract_palm_region_enhanced(image2)
    
    # Extract SIFT features
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
    kp1, des1 = sift.detectAndCompute(palm1, None)
    kp2, des2 = sift.detectAndCompute(palm2, None)
    
    # LAYER 1: SIFT Feature Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    sift_score = len(good_matches)
    
    # LAYER 2: Texture Correlation
    texture1 = calculate_texture_histogram(palm1)
    texture2 = calculate_texture_histogram(palm2)
    texture_score = np.corrcoef(texture1, texture2)[0, 1]
    if np.isnan(texture_score):
        texture_score = 0
    
    # LAYER 3: Geometric Similarity
    geo1 = calculate_geometric_features(kp1)
    geo2 = calculate_geometric_features(kp2)
    geometric_score = np.corrcoef(geo1, geo2)[0, 1]
    if np.isnan(geometric_score):
        geometric_score = 0
    
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
    tests = [
        ("SIFT Matches", sift_score, 60),
        ("Texture Correlation", texture_score, 0.75),
        ("Geometric Similarity", geometric_score, 0.70),
        ("Template Score", template_score, 0.75),
        ("SSIM Score", ssim_score, 0.55),
        ("Edge Similarity", edge_score, 0.70)
    ]
    
    passed_tests = sum(1 for _, score, threshold in tests if score >= threshold)
    
    # Composite score calculation
    final_score = (
        sift_score * 3.0 +
        texture_score * 120 +
        geometric_score * 80 +
        template_score * 100 +
        ssim_score * 150 +
        edge_score * 80
    )
    
    # Authentication decision (ALL 6 tests must pass AND high composite score)
    authenticated = (passed_tests >= 6 and final_score >= 300)
    
    # Return detailed results
    return authenticated, final_score, {
        'sift_matches': sift_score,
        'texture_correlation': texture_score,
        'geometric_similarity': geometric_score,
        'template_score': template_score,
        'ssim_score': ssim_score,
        'edge_similarity': edge_score,
        'passed_tests': passed_tests,
        'total_tests': 6,
        'final_score': final_score
    }


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
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
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


# ============================================================================
# CODE CHUNK 5: PERFORMANCE METRICS CALCULATION
# ============================================================================

def calculate_biometric_metrics(genuine_results, impostor_results):
    """
    Calculate GAR, FAR, FRR, TRR, and Accuracy
    Following ISO/IEC 19795 biometric evaluation standards
    
    Args:
        genuine_results: List of dicts with 'authenticated' key (same person tests)
        impostor_results: List of dicts with 'authenticated' key (different people tests)
    
    Returns:
        Dictionary with all calculated metrics
    """
    
    # Count genuine acceptances and rejections
    genuine_accepted = sum(1 for t in genuine_results if t['authenticated'])
    genuine_rejected = len(genuine_results) - genuine_accepted
    
    # Count impostor acceptances and rejections
    impostor_accepted = sum(1 for t in impostor_results if t['authenticated'])
    impostor_rejected = len(impostor_results) - impostor_accepted
    
    # Calculate rates (as percentages)
    GAR = (genuine_accepted / len(genuine_results)) * 100 if genuine_results else 0
    FRR = (genuine_rejected / len(genuine_results)) * 100 if genuine_results else 0
    FAR = (impostor_accepted / len(impostor_results)) * 100 if impostor_results else 0
    TRR = (impostor_rejected / len(impostor_results)) * 100 if impostor_results else 0
    
    # Overall accuracy
    total_correct = genuine_accepted + impostor_rejected
    total_tests = len(genuine_results) + len(impostor_results)
    accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
    
    metrics = {
        'GAR': GAR,        # Genuine Acceptance Rate (higher = better UX)
        'FAR': FAR,        # False Acceptance Rate (lower = better security)
        'FRR': FRR,        # False Rejection Rate (lower = better UX)
        'TRR': TRR,        # True Rejection Rate (higher = better security)
        'Accuracy': accuracy,
        'genuine_tests': len(genuine_results),
        'impostor_tests': len(impostor_results),
        'true_positives': genuine_accepted,
        'false_negatives': genuine_rejected,
        'false_positives': impostor_accepted,
        'true_negatives': impostor_rejected
    }
    
    # Print formatted results
    print(f"\n{'='*60}")
    print(f"BIOMETRIC PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Test Configuration:")
    print(f"  Genuine tests: {len(genuine_results)}")
    print(f"  Impostor tests: {len(impostor_results)}")
    print(f"  Total tests: {total_tests}")
    print(f"\nPerformance Metrics:")
    print(f"  Overall Accuracy: {accuracy:.2f}%")
    print(f"  GAR (Genuine Acceptance): {GAR:.2f}%")
    print(f"  FAR (False Acceptance): {FAR:.2f}%")
    print(f"  FRR (False Rejection): {FRR:.2f}%")
    print(f"  TRR (True Rejection): {TRR:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP): {genuine_accepted}")
    print(f"  False Negatives (FN): {genuine_rejected}")
    print(f"  False Positives (FP): {impostor_accepted}")
    print(f"  True Negatives (TN): {impostor_rejected}")
    print(f"{'='*60}\n")
    
    return metrics


# ============================================================================
# COMPLETE USAGE EXAMPLE
# ============================================================================

import cv2
import numpy as np

# Example 1: Simple comparison
palm1 = cv2.imread("palm_left_20250924_133551.bmp")
palm2 = cv2.imread("registered_palm_left.bmp")

authenticated, matches, results = compare_palms_and_return_results(palm1, palm2)

if authenticated:
    print(f"✅ Authentication SUCCESSFUL - {matches} matches found")
else:
    print(f"❌ Authentication FAILED - Only {matches} matches (need {results['match_threshold']})")

# Example 2: Six-layer validation
authenticated, score, details = six_layer_authentication(palm1, palm2)
print(f"\nSix-Layer Validation: {'AUTHENTICATED' if authenticated else 'REJECTED'}")
print(f"Final Score: {score:.2f}")
print(f"Tests Passed: {details['passed_tests']}/6")

# Example 3: Calculate metrics from test results
genuine_tests = [
    {'authenticated': True},
    {'authenticated': True},
    {'authenticated': False}
]
impostor_tests = [
    {'authenticated': False},
    {'authenticated': False},
    {'authenticated': True}
]
metrics = calculate_biometric_metrics(genuine_tests, impostor_tests)
