"""Quick test for simplified API"""
import requests
import json

BASE = "http://localhost:8001"

print("=" * 60)
print("Testing Simplified AI Service")
print("=" * 60)

# Test 1: Health
print("\n1. Health Check")
r = requests.get(f"{BASE}/health")
print(f"   Status: {r.status_code}")
print(f"   Response: {r.json()}")

# Test 2: Requirement Quality
print("\n2. Requirement Quality")
r = requests.post(
    f"{BASE}/api/inference/requirement-quality",
    json={"texts": ["The system shall be fast."]}
)
print(f"   Status: {r.status_code}")
if r.status_code == 200:
    result = r.json()
    print(f"   Success: {result['success']}")
    if result['success'] and result['predictions']:
        pred = result['predictions'][0]
        print(f"   Labels: {pred['predicted_labels']}")

# Test 3: Requirement Modification
print("\n3. Requirement Modification")
r = requests.post(
    f"{BASE}/api/inference/requirement-modification",
    json={
        "original_requirement": "Send email notifications.",
        "modification_instruction": "Use SMS instead."
    }
)
print(f"   Status: {r.status_code}")
if r.status_code == 200:
    result = r.json()
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Modified: {result['modified_requirement']}")

print("\n" + "=" * 60)
print("Tests complete!")
print("=" * 60)

