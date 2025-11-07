# SIFT Implementation: 3 Essential Code Snippets

---

## Code Snippet 1: Extract Region of Interest (ROI)

```python
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
```

**Purpose**: Extracts the palm region of interest and applies 3-stage preprocessing:
- Bilateral filtering removes noise while preserving palm line edges
- CLAHE enhances local contrast for better feature detection
- Gamma correction normalizes varying lighting conditions

**Label**: Figure X.X - Palm region extraction and preprocessing pipeline

---

## Code Snippet 2: Load Model and Extract Features from Palm

```python
def load_model_and_extract_features(palm_image):
    """
    Load SIFT model (detector) and extract features from palm image
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
```

**Purpose**: 
- Initializes SIFT detector with parameters optimized for palmprint recognition
- Extracts scale-invariant keypoints from palm region
- Generates 128-dimensional descriptors for each keypoint
- Each descriptor is invariant to rotation, scale, and illumination changes

**Returns**:
- `keypoints`: List of detected feature points with their properties (location, scale, orientation)
- `descriptors`: Numpy array (n × 128) where each row is a 128-D feature vector

**Label**: Figure X.X - SIFT feature extraction with optimized parameters

---

## Code Snippet 3: Compare Palms and Return Results

```python
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


# Example usage:
if __name__ == "__main__":
    # Load two palm images
    palm1 = cv2.imread("palm_left_20250924_133551.bmp")
    palm2 = cv2.imread("registered_palm_left.bmp")
    
    # Compare palms
    authenticated, matches, results = compare_palms_and_return_results(palm1, palm2)
    
    if authenticated:
        print(f"✅ Authentication SUCCESSFUL - {matches} matches found")
    else:
        print(f"❌ Authentication FAILED - Only {matches} matches (need {results['match_threshold']})")
```

**Purpose**: 
- Compares two palmprint images using SIFT feature matching
- Applies Lowe's ratio test to filter reliable matches
- Makes authentication decision based on match count threshold
- Returns comprehensive results including match statistics

**Process**:
1. Extract SIFT features from both images
2. Match descriptors using Brute-Force matcher with k=2
3. Apply Lowe's ratio test (0.75 threshold)
4. Count good matches
5. Compare against threshold (≥60 matches for authentication)
6. Return decision and detailed statistics

**Label**: Figure X.X - Palm comparison and authentication decision workflow

---

## Complete Working Example

```python
import cv2
import numpy as np

# Complete pipeline combining all three snippets
def authenticate_palmprint(query_image_path, registered_image_path):
    """
    Complete palmprint authentication pipeline
    """
    
    # Load images
    query_image = cv2.imread(query_image_path)
    registered_image = cv2.imread(registered_image_path)
    
    if query_image is None or registered_image is None:
        print("❌ Error: Could not load images")
        return False
    
    # Step 1: Extract ROI from both images
    print("Step 1: Extracting palm regions...")
    query_roi = extract_palm_region_enhanced(query_image)
    registered_roi = extract_palm_region_enhanced(registered_image)
    
    # Step 2: Extract features (load model and compute descriptors)
    print("Step 2: Extracting SIFT features...")
    query_kp, query_des = load_model_and_extract_features(query_image)
    registered_kp, registered_des = load_model_and_extract_features(registered_image)
    
    # Step 3: Compare and get results
    print("Step 3: Comparing palms...")
    is_match, match_count, results = compare_palms_and_return_results(
        query_image, 
        registered_image,
        ratio_threshold=0.75,
        match_threshold=60
    )
    
    return is_match, results

# Run authentication
authenticated, results = authenticate_palmprint(
    "palm_left_20250924_133551.bmp",
    "registered_palm_left.bmp"
)
```

**Label**: Figure X.X - Complete SIFT palmprint authentication pipeline

---

## Summary Table

| Snippet | Function | Input | Output | Purpose |
|---------|----------|-------|--------|---------|
| **1** | `extract_palm_region_enhanced()` | Raw image | Preprocessed ROI | Extract and enhance palm region |
| **2** | `load_model_and_extract_features()` | Palm image | Keypoints, Descriptors | Initialize SIFT and extract features |
| **3** | `compare_palms_and_return_results()` | Two palm images | Boolean, Count, Dict | Match features and authenticate |

---

## Key Parameters

```python
# SIFT Configuration
SIFT_PARAMS = {
    'nfeatures': 1200,          # Maximum keypoints to extract
    'contrastThreshold': 0.02,  # Feature detection sensitivity
    'edgeThreshold': 12,        # Edge detection threshold
    'sigma': 1.6                # Gaussian blur sigma
}

# Matching Configuration
MATCHING_PARAMS = {
    'ratio_threshold': 0.75,    # Lowe's ratio test threshold
    'match_threshold': 60       # Minimum matches for authentication
}

# Preprocessing Configuration
PREPROCESSING_PARAMS = {
    'bilateral_d': 9,           # Bilateral filter diameter
    'bilateral_sigma': 75,      # Bilateral filter sigma
    'clahe_clip': 3.0,         # CLAHE clip limit
    'clahe_grid': (8, 8)       # CLAHE tile grid size
}
```

