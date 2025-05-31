#!/usr/bin/env python3
"""
Quick test with real face image
"""
import requests
import base64
import json
from PIL import Image, ImageDraw # Added ImageDraw
import io
import os # Added os for path manipulation

# Define a path for the test image, similar to test_live_api.py
# This helps in saving and reusing the image if needed, or just for consistency
QUICK_TEST_IMAGE_PATH = "../../data/test_images/quick_test_face.jpg"

def create_minimal_test_image_for_mediapipe():
    """Create a simple image that MediaPipe should reliably detect."""
    print("🎨 Creating minimal test image for MediaPipe...")
    img = Image.new('RGB', (300, 300), 'white')
    draw = ImageDraw.Draw(img)

    # Draw a very simple, clear face. MediaPipe is good with clear features.
    # Large, dark eyes and a simple mouth on a plain background.
    
    # Head outline (optional, but can help)
    # draw.ellipse([50, 50, 250, 250], fill='lightyellow', outline='black')

    # Eyes (simple dark circles)
    eye_radius = 20
    draw.ellipse([100 - eye_radius, 100 - eye_radius, 100 + eye_radius, 100 + eye_radius], fill='black') # Left eye
    draw.ellipse([200 - eye_radius, 100 - eye_radius, 200 + eye_radius, 100 + eye_radius], fill='black') # Right eye

    # Mouth (simple line or rectangle)
    draw.line([120, 200, 180, 200], fill='black', width=10) # Simple line mouth

    # Ensure directory exists
    os.makedirs(os.path.dirname(QUICK_TEST_IMAGE_PATH), exist_ok=True)
    img.save(QUICK_TEST_IMAGE_PATH, format='JPEG')
    print(f"✅ Minimal test image saved to {QUICK_TEST_IMAGE_PATH}")
    return QUICK_TEST_IMAGE_PATH

def run_quick_mediapipe_test():
    """Test with a minimal approach using MediaPipe."""
    print("\n🧪 Quick MediaPipe test...")
    
    # Create and get path to the test image
    image_path = create_minimal_test_image_for_mediapipe()
    
    try:
        with open(image_path, 'rb') as img_file:
            # Read the file content for the request
            img_file_content = img_file.read()

        # Test API
        # The `files` parameter expects a file-like object or (filename, file_content, content_type)
        response = requests.post(
            'http://localhost:8000/api/v1/detect', 
            files={'image': ('quick_test.jpg', img_file_content, 'image/jpeg')},
            data={'detector': 'mediapipe'}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Faces found: {len(result.get('faces', []))}")
                if len(result.get('faces', [])) > 0:
                    print("✅ MediaPipe detected a face in the minimal image.")
                else:
                    print("⚠️ MediaPipe did NOT detect a face in the minimal image.")
            except json.JSONDecodeError:
                print("❌ Error: Could not decode JSON response.")
                print(f"Raw response: {response.text}")
        else:
            print(f"❌ Error: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ Error: Test image not found at {image_path}")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server. Is it running?")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_quick_mediapipe_test()
