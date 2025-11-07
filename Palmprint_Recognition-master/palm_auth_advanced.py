import cv2
import numpy as np
from SIFT_DIP import get_sift_features, sift_detect_match_num
from datetime import datetime

def extract_palm_region(img):
    """Extract and enhance the palm region for more focused comparison"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    height, width = gray.shape
    
    # Extract larger center region
    center_x, center_y = width // 2, height // 2
    region_size = min(width, height) // 2  # Larger region
    
    palm_region = gray[max(0, center_y-region_size):min(height, center_y+region_size), 
                      max(0, center_x-region_size):min(width, center_x+region_size)]
    
    # Enhance the palm region for better feature detection
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(palm_region)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return filtered

def calculate_texture_histogram(img):
    """Calculate texture-based features using histogram analysis"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Calculate histogram of oriented gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    
    # Create histogram
    hist, _ = np.histogram(magnitude.ravel(), bins=64, range=(0, 255))
    hist = hist.astype(float)
    hist = hist / (hist.sum() + 1e-7)
    return hist

def calculate_geometric_features(keypoints):
    """Calculate geometric properties of keypoints distribution"""
    if len(keypoints) < 5:
        return np.array([0, 0, 0, 0, 0, 0])
    
    points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    # Calculate center and spread
    center = np.mean(points, axis=0)
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    
    # Calculate angles from center
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    features = [
        np.mean(distances),      # Average distance from center
        np.std(distances),       # Standard deviation of distances
        len(keypoints),          # Number of keypoints
        np.std(angles),          # Angular distribution
        np.max(distances) - np.min(distances),  # Range of distances
        np.mean([kp.response for kp in keypoints])  # Average response strength
    ]
    
    return np.array(features)

def advanced_authenticate_palm(captured_img, registered_img):
    """Advanced multi-layer palm authentication"""
    
    # Extract focused palm regions
    palm1 = extract_palm_region(captured_img)
    palm2 = extract_palm_region(registered_img)
    
    if palm1.size == 0 or palm2.size == 0:
        return False, 0, "Could not extract palm regions"
    
    # Check if palm regions are reasonably similar in size
    size_ratio = min(palm1.size, palm2.size) / max(palm1.size, palm2.size)
    if size_ratio < 0.7:  # If size difference is more than 30%
        return False, 0, f"Palm region size mismatch: ratio {size_ratio:.2f}"
    
    # Preprocess both images
    from SIFT_DIP import get_sift_features  # Import here to avoid circular import
    
    # Enhanced SIFT with more sensitive parameters for better feature detection
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03, edgeThreshold=10)
    
    # Get keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(palm1, None)
    kp2, des2 = sift.detectAndCompute(palm2, None)
    
    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return False, 0, f"Insufficient features detected: {len(des1) if des1 is not None else 0} vs {len(des2) if des2 is not None else 0}"
    
    # Layer 1: SIFT matching with very strict ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.2 * n.distance:  # Extremely strict ratio
                good_matches.append(m)
    
    # Additional validation: Check for spatial consistency of matches
    if len(good_matches) >= 10:
        # Use RANSAC to find homography and filter out inconsistent matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        try:
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, 
                                                ransacReprojThreshold=5.0)
            if homography is not None:
                # Count inliers (spatially consistent matches)
                inliers = np.sum(mask)
                sift_score = inliers
            else:
                sift_score = 0
        except:
            sift_score = 0
    else:
        sift_score = len(good_matches)
    
    # Layer 2: Texture histogram comparison
    hist1 = calculate_texture_histogram(palm1)
    hist2 = calculate_texture_histogram(palm2)
    texture_correlation = cv2.compareHist(hist1.astype(np.float32), 
                                         hist2.astype(np.float32), 
                                         cv2.HISTCMP_CORREL)
    
    # Layer 3: Geometric feature comparison
    geom1 = calculate_geometric_features(kp1)
    geom2 = calculate_geometric_features(kp2)
    
    # Normalize geometric features
    geom1_norm = geom1 / (np.linalg.norm(geom1) + 1e-7)
    geom2_norm = geom2 / (np.linalg.norm(geom2) + 1e-7)
    geom_similarity = np.dot(geom1_norm, geom2_norm)
    
    # Layer 4: Template matching with normalized cross-correlation
    # Resize to same size for template matching
    h, w = min(palm1.shape[0], palm2.shape[0]), min(palm1.shape[1], palm2.shape[1])
    palm1_resized = cv2.resize(palm1, (w, h))
    palm2_resized = cv2.resize(palm2, (w, h))
    
    template_result = cv2.matchTemplate(palm1_resized, palm2_resized, cv2.TM_CCOEFF_NORMED)
    template_score = np.max(template_result)
    
    # Layer 5: Structural Similarity Index (SSIM)
    def calculate_ssim(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        ssim_val = ((2*mu1*mu2 + c1) * (2*cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        return max(0, ssim_val)  # Ensure non-negative
    
    ssim_score = calculate_ssim(palm1_resized, palm2_resized)
    
    # Layer 6: Edge comparison using Canny
    edges1 = cv2.Canny(palm1_resized, 50, 150)
    edges2 = cv2.Canny(palm2_resized, 50, 150)
    
    # Calculate edge similarity
    edge_diff = cv2.absdiff(edges1, edges2)
    edge_similarity = 1.0 - (np.sum(edge_diff) / (255.0 * edge_diff.size))
    
    # Combine all scores with strict weights
    final_score = (
        sift_score * 2.0 +           # SIFT matches (highest weight)
        texture_correlation * 50 +   # Texture correlation
        geom_similarity * 30 +       # Geometric similarity  
        template_score * 40 +        # Template matching
        ssim_score * 35 +           # Structural similarity
        edge_similarity * 25        # Edge similarity
    )
    
    # Individual component thresholds (ALL must pass)
    strict_thresholds = {
        'sift_matches': (sift_score, 50),           # At least 50 good matches
        'texture_corr': (texture_correlation, 0.7), # At least 70% texture correlation
        'geometric': (geom_similarity, 0.6),        # At least 60% geometric similarity
        'template': (template_score, 0.4),          # At least 40% template match
        'ssim': (ssim_score, 0.5),                 # At least 50% structural similarity
        'edges': (edge_similarity, 0.7)            # At least 70% edge similarity
    }
    
    # Check which tests pass
    passed_tests = []
    failed_tests = []
    
    for test_name, (score, threshold) in strict_thresholds.items():
        if score >= threshold:
            passed_tests.append(test_name)
        else:
            failed_tests.append((test_name, score, threshold))
    
    # Detailed results
    details = {
        'sift_matches': sift_score,
        'texture_correlation': texture_correlation,
        'geometric_similarity': geom_similarity,
        'template_score': template_score,
        'ssim_score': ssim_score,
        'edge_similarity': edge_similarity,
        'final_score': final_score,
        'passed_tests': len(passed_tests),
        'total_tests': len(strict_thresholds),
        'failed_tests': failed_tests
    }
    
    # Authentication passes ONLY if ALL tests pass AND final score > 300
    is_authentic = len(passed_tests) == len(strict_thresholds) and final_score > 300
    
    return is_authentic, final_score, details

def capture_palm():
    """Capture palm image from webcam with real-time quality indicators"""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return None
    
    # Set higher resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera opened. Follow the on-screen indicators for best quality...")
    
    while True:
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            guide_size = min(width, height) // 3
            
            # Draw main guide circle
            cv2.circle(frame, center, guide_size, (0, 255, 0), 3)
            
            # Draw crosshair for centering
            cv2.line(frame, (center[0]-20, center[1]), (center[0]+20, center[1]), (0, 255, 0), 2)
            cv2.line(frame, (center[0], center[1]-20), (center[0], center[1]+20), (0, 255, 0), 2)
            
            # Extract ROI for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_y1 = max(0, center[1] - guide_size)
            roi_y2 = min(height, center[1] + guide_size)
            roi_x1 = max(0, center[0] - guide_size)
            roi_x2 = min(width, center[0] + guide_size)
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Quality indicators
            indicators = []
            overall_quality = True
            
            if roi.size > 0:
                # 1. Brightness check
                avg_brightness = np.mean(roi)
                if avg_brightness < 60:
                    indicators.append(("LIGHTING: Too Dark", (0, 0, 255)))
                    overall_quality = False
                elif avg_brightness > 200:
                    indicators.append(("LIGHTING: Too Bright", (0, 0, 255)))
                    overall_quality = False
                else:
                    indicators.append(("LIGHTING: Good", (0, 255, 0)))
                
                # 2. Contrast check
                contrast = np.std(roi)
                if contrast < 25:
                    indicators.append(("CONTRAST: Too Low", (0, 0, 255)))
                    overall_quality = False
                elif contrast > 85:
                    indicators.append(("CONTRAST: Too High", (0, 0, 255)))
                    overall_quality = False
                else:
                    indicators.append(("CONTRAST: Good", (0, 255, 0)))
                
                # 3. Sharpness check (using Laplacian variance)
                laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
                if laplacian_var < 50:
                    indicators.append(("SHARPNESS: Blurry", (0, 0, 255)))
                    overall_quality = False
                else:
                    indicators.append(("SHARPNESS: Good", (0, 255, 0)))
                
                # 4. Feature density check
                # Quick SIFT check for feature richness
                sift = cv2.SIFT_create(nfeatures=100)
                keypoints = sift.detect(roi, None)
                feature_count = len(keypoints)
                
                if feature_count < 10:
                    indicators.append(("FEATURES: Too Few", (0, 0, 255)))
                    overall_quality = False
                else:
                    indicators.append(("FEATURES: Sufficient", (0, 255, 0)))
                
                # 5. Movement check (using frame difference)
                # This would require storing previous frame, simplified for now
                indicators.append(("STABILITY: Hold Steady", (255, 255, 0)))  # Yellow reminder
            
            # Draw quality indicators with background
            indicator_y = 50
            for i, (text, color) in enumerate(indicators):
                # Background rectangle for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (20, indicator_y - 25), (20 + text_size[0] + 10, indicator_y + 10), (0, 0, 0), -1)
                cv2.putText(frame, text, (25, indicator_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                indicator_y += 40
            
            # Overall status
            if overall_quality:
                status_text = "READY TO CAPTURE - Press SPACE"
                status_color = (0, 255, 0)
                # Make the guide circle brighter when ready
                cv2.circle(frame, center, guide_size, (0, 255, 0), 5)
            else:
                status_text = "ADJUST POSITION/LIGHTING"
                status_color = (0, 0, 255)
            
            # Status bar at bottom
            status_bg_height = 60
            cv2.rectangle(frame, (0, height - status_bg_height), (width, height), (0, 0, 0), -1)
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(frame, status_text, (text_x, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Instructions
            cv2.putText(frame, "Q to quit", (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Quality score
            quality_score = sum(1 for _, color in indicators[:4] if color == (0, 255, 0))  # Count green indicators
            cv2.putText(frame, f"Quality: {quality_score}/4", (width - 150, height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Palm Capture - Quality Guide', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if overall_quality:
                    print("\nCapturing high-quality image...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame
                else:
                    print("\nPlease improve image quality before capturing (see indicators)")
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    while True:
        print("\n=== Advanced Palm Authentication System ===")
        print("1. Register New Palm")
        print("2. Authenticate Palm")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1':
            print("\nCapturing palm for registration...")
            palm_type = input("Enter palm type (left/right): ").lower()
            while palm_type not in ['left', 'right']:
                palm_type = input("Please enter 'left' or 'right': ").lower()
            
            img = capture_palm()
            if img is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"registered_palm_{palm_type}.bmp"
                
                cv2.imwrite(filename, img)
                cv2.imwrite(f"palm_{palm_type}_{timestamp}.bmp", img)
                
                with open("palm_info.txt", "w") as f:
                    f.write(f"Type: {palm_type}\nRegistered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"\n{palm_type.capitalize()} palm registered successfully!")
            else:
                print("Registration failed!")
        
        elif choice == '2':
            try:
                # Try to load registered palm
                registered_palm = None
                palm_type = "unknown"
                
                try:
                    with open("palm_info.txt", "r") as f:
                        info = f.read()
                        if "Type:" in info:
                            palm_type = info.split("Type:")[1].split("\n")[0].strip()
                            registered_palm = cv2.imread(f"registered_palm_{palm_type}.bmp")
                except:
                    pass
                
                if registered_palm is None:
                    print("\nNo registered palm found. Please register first.")
                    continue
                
                print(f"\nPlace your {palm_type} palm for authentication...")
                captured_img = capture_palm()
                
                if captured_img is not None:
                    is_match, score, details = advanced_authenticate_palm(captured_img, registered_palm)
                    
                    print(f"\n{'='*50}")
                    print("AUTHENTICATION RESULTS")
                    print(f"{'='*50}")
                    
                    if isinstance(details, str):
                        print(f"Error: {details}")
                    else:
                        print(f"SIFT Matches: {details['sift_matches']}")
                        print(f"Texture Correlation: {details['texture_correlation']:.3f}")
                        print(f"Geometric Similarity: {details['geometric_similarity']:.3f}")
                        print(f"Template Score: {details['template_score']:.3f}")
                        print(f"SSIM Score: {details['ssim_score']:.3f}")
                        print(f"Edge Similarity: {details['edge_similarity']:.3f}")
                        print(f"Final Score: {details['final_score']:.2f}")
                        print(f"Tests Passed: {details['passed_tests']}/{details['total_tests']}")
                        
                        if details['failed_tests']:
                            print("\nFailed Tests:")
                            for test_name, score, threshold in details['failed_tests']:
                                print(f"  {test_name}: {score:.3f} < {threshold}")
                    
                    print(f"\nResult: {'AUTHENTICATED' if is_match else 'REJECTED'}")
                    print(f"{'='*50}")
                    
            except Exception as e:
                print(f"\nError during authentication: {str(e)}")
        
        elif choice == '3':
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()