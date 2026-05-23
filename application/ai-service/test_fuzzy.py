"""Test script for fuzzy inference system endpoints"""
import requests
import json

BASE = "http://localhost:8001"

print("=" * 80)
print("Testing Fuzzy Inference Systems")
print("=" * 80)

# Test 1: Defect Severity
print("\n1. Defect Severity FIS")
print("-" * 80)
payload = {
    "subjective": 0.3,
    "ambiguous": 0.7,
    "nonverifiable": 0.6,
    "negative": 0.2,
    "vague": 0.8
}
print(f"Inputs: {json.dumps(payload, indent=2)}")

try:
    r = requests.post(f"{BASE}/api/fuzzy/defect-severity", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Defect Severity: {result['defect_severity']:.4f}")
            print(f"Label: {result['defect_severity_label']}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Correctness
print("\n2. Correctness FIS")
print("-" * 80)
payload = {
    "preservation_correctness": 0.8,
    "change_correctness": 0.7
}
print(f"Inputs: {json.dumps(payload, indent=2)}")

try:
    r = requests.post(f"{BASE}/api/fuzzy/correctness", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Correctness: {result['correctness']:.4f}")
            print(f"Label: {result['correctness_label']}")
            correctness_value = result['correctness']
except Exception as e:
    print(f"Error: {e}")

# Test 3: Requirement Quality (using previous results)
print("\n3. Requirement Quality FIS")
print("-" * 80)
# Use defect_severity from Test 1 and correctness from Test 2
payload = {
    "defect_severity": 0.65,  # High defects
    "correctness": 0.75       # High correctness
}
print(f"Inputs: {json.dumps(payload, indent=2)}")

try:
    r = requests.post(f"{BASE}/api/fuzzy/requirement-quality", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Requirement Quality: {result['requirement_quality']:.4f}")
            print(f"Label: {result['requirement_quality_label']}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Complete workflow
print("\n4. Complete Workflow Example")
print("-" * 80)
print("Step 1: Calculate Defect Severity")
defect_payload = {
    "subjective": 0.2,
    "ambiguous": 0.3,
    "nonverifiable": 0.25,
    "negative": 0.15,
    "vague": 0.2
}
r1 = requests.post(f"{BASE}/api/fuzzy/defect-severity", json=defect_payload)
defect_result = r1.json()
print(f"  Defect Severity: {defect_result.get('defect_severity', 0):.4f} ({defect_result.get('defect_severity_label', 'N/A')})")

print("\nStep 2: Calculate Correctness")
correctness_payload = {
    "preservation_correctness": 0.85,
    "change_correctness": 0.8
}
r2 = requests.post(f"{BASE}/api/fuzzy/correctness", json=correctness_payload)
correctness_result = r2.json()
print(f"  Correctness: {correctness_result.get('correctness', 0):.4f} ({correctness_result.get('correctness_label', 'N/A')})")

print("\nStep 3: Calculate Overall Quality")
quality_payload = {
    "defect_severity": defect_result.get('defect_severity', 0.5),
    "correctness": correctness_result.get('correctness', 0.5)
}
r3 = requests.post(f"{BASE}/api/fuzzy/requirement-quality", json=quality_payload)
quality_result = r3.json()
print(f"  Requirement Quality: {quality_result.get('requirement_quality', 0):.4f} ({quality_result.get('requirement_quality_label', 'N/A')})")

print("\n" + "=" * 80)
print("Interpretation:")
print("-" * 80)
print(f"With low defect severity ({defect_result.get('defect_severity', 0):.2f}) and")
print(f"high correctness ({correctness_result.get('correctness', 0):.2f}),")
print(f"the overall requirement quality is: {quality_result.get('requirement_quality_label', 'N/A').upper()}")
print(f"(Score: {quality_result.get('requirement_quality', 0):.4f})")
print("=" * 80)

