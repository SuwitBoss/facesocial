#!/usr/bin/env python3
"""
Test script for live FastAPI endpoints
Tests the face detection service API endpoints with real HTTP requests
"""
import requests
import json
import time
import os
from PIL import Image, ImageDraw
import io
import base64

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "../../data/test_images/test_face.jpg"

def create_better_test_image():
    """Create a more realistic test face image"""
    print("\n🎨 Creating better test image...")
    
    # Create a larger and more detailed image
    img = Image.new('RGB', (400, 400), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # Draw a more realistic face
    # Head (oval)
    draw.ellipse([100, 80, 300, 280], fill='#FFDBAC', outline='#CD853F', width=3)
    
    # Eyes (left, right)
    draw.ellipse([130, 140, 160, 170], fill='white', outline='black', width=2)
    draw.ellipse([145, 150, 155, 165], fill='black')  # Pupils
    
    draw.ellipse([240, 140, 270, 170], fill='white', outline='black', width=2)
    draw.ellipse([250, 150, 260, 165], fill='black')  # Pupils
    
    # Eyebrows
    draw.arc([125, 120, 165, 140], start=0, end=180, fill='#8B4513', width=4)
    draw.arc([235, 120, 275, 140], start=0, end=180, fill='#8B4513', width=4)
    
    # Nose
    draw.polygon([(200, 180), (190, 210), (200, 215), (210, 210)], fill='#FFCDA3')
    
    # Mouth
    draw.ellipse([170, 230, 230, 250], fill='#FF6B6B', outline='#CC5555', width=2)
    
    # Hair
    draw.arc([90, 60, 310, 180], start=180, end=360, fill='#8B4513', width=20)
    
    # Save the image
    os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)
    img.save(TEST_IMAGE_PATH)
    print(f"✅ Better test image created: {TEST_IMAGE_PATH}")
    return True

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n📊 Testing root endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Service: {data.get('service', 'Unknown')}")
        print(f"Version: {data.get('version', 'Unknown')}")
        print(f"Available detectors: {data.get('available_detectors', [])}")
        print(f"Available modes: {data.get('modes', [])}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint check failed: {e}")
        return False

def test_detection_endpoint(detector_type):
    """Test face detection endpoint"""
    print(f"\n🔍 Testing {detector_type} detection endpoint...")
    
    # Check if test image exists, create if not
    if not os.path.exists(TEST_IMAGE_PATH):
        create_better_test_image() # ✅ Changed from create_test_image()
    
    try:
        # Prepare the image file
        with open(TEST_IMAGE_PATH, 'rb') as img_file:
            files = {'image': ('test_face.jpg', img_file, 'image/jpeg')}
            data = {'detector': detector_type}
            
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/api/v1/detect", files=files, data=data)
            end_time = time.time()
            
            print(f"Status Code: {response.status_code}")
            print(f"Processing time: {end_time - start_time:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Faces detected: {len(result.get('faces', []))}")
                
                for i, face in enumerate(result.get('faces', [])):
                    bbox = face.get('bbox', {})
                    confidence = face.get('confidence', 0)
                    print(f"  Face {i+1}: confidence={confidence:.3f}, bbox=({bbox.get('x', 0)}, {bbox.get('y', 0)}, {bbox.get('width', 0)}, {bbox.get('height', 0)})")
                
                return True
            else:
                print(f"❌ Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_batch_detection():
    """Test batch detection endpoint"""
    print(f"\n📦 Testing batch detection endpoint...")
    
    # Check if test image exists, create if not
    if not os.path.exists(TEST_IMAGE_PATH):
        create_better_test_image() # ✅ Changed from create_test_image()
    
    try:
        # Prepare multiple files (using same image for simplicity)
        files = []
        with open(TEST_IMAGE_PATH, 'rb') as img_file:
            img_data = img_file.read()
            
        files.append(('images', ('test1.jpg', io.BytesIO(img_data), 'image/jpeg')))
        files.append(('images', ('test2.jpg', io.BytesIO(img_data), 'image/jpeg')))
        
        data = {'detector': 'mediapipe'}  # Use fastest detector for batch test
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/api/v1/detect/batch", files=files, data=data)
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Processing time: {end_time - start_time:.3f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Images processed: {len(result.get('results', []))}")
            
            for i, img_result in enumerate(result.get('results', [])):
                faces = img_result.get('faces', [])
                print(f"  Image {i+1}: {len(faces)} faces detected")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch detection test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting Live API Tests")
    print("=" * 50)
    
    test_results = []
      # Test basic endpoints
    test_results.append(("Health Check", test_health_endpoint()))
    test_results.append(("Root Endpoint", test_root_endpoint()))
    
    # Test detection endpoints for each detector
    detectors = ['yolo', 'insightface', 'mediapipe']
    for detector in detectors:
        test_results.append((f"{detector.upper()} Detection", test_detection_endpoint(detector)))
    
    # Test batch detection
    test_results.append(("Batch Detection", test_batch_detection()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Face Detection Service is fully functional!")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
