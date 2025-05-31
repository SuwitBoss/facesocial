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

def create_test_image():
    """Create a simple test image with a face-like shape"""
    print("\n🎨 Creating test image...")
    
    # Create a simple image with a basic face-like shape
    img = Image.new('RGB', (200, 200), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    # Head (circle)
    draw.ellipse([50, 50, 150, 150], fill='peachpuff', outline='black', width=2)
    
    # Eyes
    draw.ellipse([70, 80, 85, 95], fill='black')
    draw.ellipse([115, 80, 130, 95], fill='black')
    
    # Nose
    draw.polygon([(100, 100), (95, 115), (105, 115)], fill='pink')
    
    # Mouth
    draw.arc([80, 115, 120, 135], start=0, end=180, fill='red', width=3)
    
    # Save the image
    os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)
    img.save(TEST_IMAGE_PATH)
    print(f"✅ Test image created: {TEST_IMAGE_PATH}")
    return True

def test_detection_endpoint(detector_type):
    """Test face detection endpoint"""
    print(f"\n🔍 Testing {detector_type} detection endpoint...")
    
    # Check if test image exists, create if not
    if not os.path.exists(TEST_IMAGE_PATH):
        create_test_image()
    
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
        create_test_image()
    
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
    detectors = ['yolo', 'mtcnn', 'mediapipe']
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
