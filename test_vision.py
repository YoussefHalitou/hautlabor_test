#!/usr/bin/env python3
"""Test script for OpenAI Vision API functionality"""

import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

def test_vision_api():
    """Test OpenAI Vision API with a simple prompt"""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    print("✅ API key found")
    
    # Test simple text completion first
    try:
        client = OpenAI(api_key=api_key)
        
        print("🧪 Testing basic OpenAI connection...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, respond with 'API works!'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ Basic API test: {result}")
        
        # Test vision capability with a simple prompt
        print("🧪 Testing Vision API availability...")
        
        # Create a simple 1x1 pixel test image
        test_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA60e6kgAAAABJRU5ErkJggg=="
        
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image? Respond in German."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{test_image_data}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        vision_result = vision_response.choices[0].message.content
        print(f"✅ Vision API test: {vision_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("🚀 Testing OpenAI Vision API...")
    success = test_vision_api()
    
    if success:
        print("\n✅ All tests passed! Vision API should work in your app.")
    else:
        print("\n❌ Tests failed. Please check your API key and OpenAI account.")
