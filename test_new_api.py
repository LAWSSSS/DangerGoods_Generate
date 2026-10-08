#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试新 API Key 是否有效"""

import requests
import json

NEW_API_KEY = "sk-sp-H.PPLML.5iiS.MEQCIFZ2BBw5HbyHaUX1_aTC6xSSiZiLlRZRultBwlbA56skAiAq2bVCmpPkhF3XjC20ZV_RMKLPwH_g3lNJHj9z6G54yg"
ENDPOINT = "http://token-plan.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

headers = {
    "Authorization": f"Bearer {NEW_API_KEY}",
    "Content-Type": "application/json",
}

# 测试图片生成
payload = {
    "model": "wan2.7-image",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "生成一张测试图片，一个红色的苹果"}
                ]
            }
        ]
    },
    "parameters": {
        "output_format": "png",
        "watermark": False,
        "size": "512*512"
    }
}

print(f"测试新 API Key: {NEW_API_KEY[:30]}...")
print("发送请求...")

try:
    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ API Key 有效！")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2)[:500]}")
    elif response.status_code == 401:
        print("❌ API Key 无效或已过期！")
    elif response.status_code == 403:
        print("❌ 无权限访问！")
    else:
        print(f"⚠️ 响应内容: {response.text[:500]}")
        
except requests.exceptions.Timeout:
    print("❌ 请求超时")
except Exception as e:
    print(f"❌ 错误: {e}")