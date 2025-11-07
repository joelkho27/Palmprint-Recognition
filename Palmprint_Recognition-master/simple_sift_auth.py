"""
Enhanced SIFT Palmprint Authentication using existing SIFT_DIP module

This system builds upon your existing SIFT_DIP.py to create a robust
palmprint authentication system with multi-template enrollment and 
advanced matching techniques.
"""

import cv2
import numpy as np
from datetime import datetime
import os
import json
from SIFT_DIP import get_sift_features, sift_detect_match_num
import warnings
warnings.filterwarnings('ignore')

class SimpleSIFTAuth:
    """Simple SIFT-based authentication using existing SIFT_DIP functions"""
    
    def __init__(self):
        self.enrolled_templates = {}
        
        # Quality thresholds
        self.min_brightness = 50
        self.max_brightness = 200
        self.min_contrast = 30
        self.min_sharpness = 100
        
        # Authentication thresholds  
        self.min_matches = 40  # Minimum SIFT matches required
        self.ratio_threshold = 0.7  # Lowe's ratio test threshold
        
        print("Simple SIFT Authentication System Ready")
        print(f"Minimum matches required: {self.min_matches}")
    
    def preprocess_palm(self, image):
        """Enhanced preprocessing for palmprint"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Normalization
        normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def assess_quality(self, image):
        """Quick quality assessment"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Laplacian for sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # SIFT features count using existing function
        processed = self.preprocess_palm(image)
        try:
            features, descriptors = get_sift_features(processed)
            feature_count = len(features) if features else 0
        except:
            feature_count = 0
        
        quality_score = (
            (100 - abs(brightness - 125) * 0.8) * 0.3 +  # Brightness
            min(100, contrast * 2) * 0.3 +               # Contrast  
            min(100, sharpness / 10) * 0.2 +             # Sharpness
            min(100, feature_count / 3) * 0.2            # Features
        )
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'feature_count': feature_count,
            'quality_score': quality_score,
            'is_good': quality_score > 70 and feature_count > 50
        }
    
    def enroll_user(self, user_id, palm_images):
        """Enroll user with multiple templates"""
        print(f"\nEnrolling user {user_id}...")
        
        if not isinstance(palm_images, list):
            palm_images = [palm_images]
        
        templates = []
        
        for i, image in enumerate(palm_images):
            print(f"Processing template {i+1}/{len(palm_images)}...")
            
            # Quality check
            quality = self.assess_quality(image)
            print(f"  Quality: {quality['quality_score']:.1f}%, Features: {quality['feature_count']}")
            
            if not quality['is_good']:
                print(f"  Template {i+1} quality too low, skipping...")
                continue
            
            # Preprocess and extract SIFT features
            processed = self.preprocess_palm(image)
            
            try:
                features, descriptors = get_sift_features(processed)
                
                if features and descriptors is not None and len(features) > 50:
                    template = {
                        'features': features,
                        'descriptors': descriptors,
                        'processed_image': processed,
                        'quality': quality,
                        'timestamp': datetime.now().isoformat()
                    }
                    templates.append(template)
                    print(f"  Template {i+1}: {len(features)} SIFT features extracted")
                else:
                    print(f"  Template {i+1}: Insufficient SIFT features")
                    
            except Exception as e:
                print(f"  Template {i+1}: SIFT extraction failed - {e}")
        
        if not templates:
            print("ERROR: No valid templates created!")
            return False
        
        # Store templates
        self.enrolled_templates[user_id] = {
            'templates': templates,
            'enrollment_date': datetime.now().isoformat(),
            'total_templates': len(templates)
        }
        
        print(f"User {user_id} enrolled with {len(templates)} templates!")
        
        # Save enrollment data
        self.save_enrollment(user_id)
        return True
    
    def authenticate_user(self, user_id, test_image):
        """Authenticate user against enrolled templates"""
        if user_id not in self.enrolled_templates:
            return False, {"error": f"User {user_id} not enrolled"}
        
        print(f"\nAuthenticating user {user_id}...")
        
        # Quality check
        quality = self.assess_quality(test_image)
        print(f"Test image quality: {quality['quality_score']:.1f}%")
        
        if not quality['is_good']:
            return False, {
                "result": "REJECTED",
                "reason": "Poor test image quality",
                "quality": quality['quality_score']
            }
        
        # Extract test features
        processed_test = self.preprocess_palm(test_image)
        
        try:
            test_features, test_descriptors = get_sift_features(processed_test)
            
            if not test_features or test_descriptors is None:
                return False, {
                    "result": "REJECTED",
                    "reason": "No SIFT features in test image"
                }
                
        except Exception as e:
            return False, {
                "result": "REJECTED", 
                "reason": f"SIFT extraction failed: {e}"
            }
        
        # Match against all templates
        user_data = self.enrolled_templates[user_id]
        best_matches = 0
        template_results = []
        
        for i, template in enumerate(user_data['templates']):
            template_descriptors = template['descriptors']
            
            try:
                # Use existing SIFT matching function
                match_count = sift_detect_match_num(
                    template_descriptors, 
                    test_descriptors, 
                    ratio=self.ratio_threshold
                )
                
                template_results.append(match_count)
                best_matches = max(best_matches, match_count)
                
                print(f"Template {i+1}: {match_count} matches")
                
            except Exception as e:
                print(f"Template {i+1}: Matching failed - {e}")
                template_results.append(0)
        
        # Authentication decision
        is_authenticated = best_matches >= self.min_matches
        
        result = {
            "authenticated": is_authenticated,
            "user_id": user_id,
            "best_matches": best_matches,
            "template_matches": template_results,
            "test_features": len(test_features),
            "quality": quality,
            "threshold": self.min_matches,
            "timestamp": datetime.now().isoformat()
        }
        
        status = "AUTHENTICATED" if is_authenticated else "REJECTED"
        print(f"Result: {status}")
        print(f"Best matches: {best_matches} (threshold: {self.min_matches})")
        
        return is_authenticated, result
    
    def save_enrollment(self, user_id):
        """Save enrollment data to file"""
        try:
            # Prepare data (can't directly save OpenCV objects)
            save_data = {
                'user_id': user_id,
                'enrollment_date': self.enrolled_templates[user_id]['enrollment_date'],
                'total_templates': self.enrolled_templates[user_id]['total_templates'],
                'template_count': len(self.enrolled_templates[user_id]['templates'])
            }
            
            filename = f'simple_sift_user_{user_id}.json'
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Enrollment data saved to {filename}")
            
        except Exception as e:
            print(f"Error saving enrollment: {e}")
    
    def capture_and_authenticate(self):
        """Real-time capture and authentication"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("\nSimple SIFT Palmprint Authentication")
        print("Commands:")
        print("  '1-9' - Enroll user ID")
        print("  'a' - Authenticate (enter user ID)")
        print("  's' - Show enrolled users")
        print("  'q' - Quit")
        
        enrollment_frames = []
        enrolling = False
        target_user = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Quality assessment
            quality = self.assess_quality(frame)
            
            # Draw UI
            display_frame = frame.copy()
            
            # Quality indicator
            bar_width = int(250 * quality['quality_score'] / 100)
            color = (0, 255, 0) if quality['is_good'] else (0, 165, 255)
            cv2.rectangle(display_frame, (20, 20), (270, 45), (0, 0, 0), 2)
            cv2.rectangle(display_frame, (20, 20), (20 + bar_width, 45), color, -1)
            
            # Text overlays
            cv2.putText(display_frame, f"Quality: {quality['quality_score']:.1f}%", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Features: {quality['feature_count']}", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if enrolling:
                cv2.putText(display_frame, f"Enrolling User {target_user}: {len(enrollment_frames)}/3", 
                           (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "Press SPACE to capture", 
                           (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(display_frame, "1-9:Enroll | A:Auth | S:Show | Q:Quit", 
                       (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('Simple SIFT Palm Auth', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle enrollment
            if key >= ord('1') and key <= ord('9') and not enrolling:
                target_user = key - ord('0')
                enrolling = True
                enrollment_frames = []
                print(f"Starting enrollment for user {target_user} - need 3 frames")
            
            elif key == ord(' ') and enrolling:
                if quality['is_good']:
                    enrollment_frames.append(frame.copy())
                    print(f"Captured frame {len(enrollment_frames)}/3")
                    
                    if len(enrollment_frames) >= 3:
                        success = self.enroll_user(target_user, enrollment_frames)
                        enrolling = False
                        enrollment_frames = []
                        target_user = None
                else:
                    print("Quality too low - try again")
            
            elif key == ord('a') and not enrolling:
                if quality['is_good']:
                    print("Enter user ID (1-9): ", end='')
                    try:
                        auth_id = int(input())
                        if 1 <= auth_id <= 9:
                            authenticated, result = self.authenticate_user(auth_id, frame)
                            print("✓ SUCCESS" if authenticated else "✗ FAILED")
                        else:
                            print("Invalid user ID")
                    except:
                        print("Invalid input")
                else:
                    print("Quality too low for authentication")
            
            elif key == ord('s'):
                enrolled = list(self.enrolled_templates.keys())
                print(f"Enrolled users: {enrolled}")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("Simple SIFT Palmprint Authentication")
    print("Uses existing SIFT_DIP.py functions")
    print("=" * 40)
    
    auth = SimpleSIFTAuth()
    auth.capture_and_authenticate()

if __name__ == "__main__":
    main()