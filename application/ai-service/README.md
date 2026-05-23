# AI Inference Service

Simple FastAPI application with two inference endpoints for requirement analysis.

## 🚀 Quick Start

```bash
# 1. Navigate to ai-service folder
cd ai-service

# 2. Create virtual environment (first time only)
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install dependencies (first time only)
pip install -r requirements.txt

# 5. Run the service
python run_simple.py
```

**Service runs at:** http://localhost:8001

**API Documentation:** http://localhost:8001/docs

## 📡 Endpoints

### Machine Learning Models

#### 1️⃣ Requirement Quality Analysis (ML)
Analyzes software requirements for quality issues using SetFit multi-label classification.

**Endpoint:** `POST /api/inference/requirement-quality`

**Request:**
```json
{
  "texts": ["The system shall be fast."]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [{
    "text": "The system shall be fast.",
    "predicted_labels": ["ambiguous", "non-verifiable"],
    "probabilities": {
      "ambiguous": 0.87,
      "non-verifiable": 0.92
    }
  }]
}
```

#### 2️⃣ Requirement Modification (ML)
Generates modified requirements based on instructions using fine-tuned Qwen model.

**Endpoint:** `POST /api/inference/requirement-modification`

**Request:**
```json
{
  "original_requirement": "Send email notifications.",
  "modification_instruction": "Use SMS instead."
}
```

**Response:**
```json
{
  "success": true,
  "modified_requirement": "Send SMS notifications."
}
```

### Fuzzy Inference Systems

#### 3️⃣ Defect Severity (Fuzzy)
Calculate defect severity based on multiple quality metrics.

**Endpoint:** `POST /api/fuzzy/defect-severity`

**Request:**
```json
{
  "subjective": 0.3,
  "ambiguous": 0.7,
  "nonverifiable": 0.6,
  "negative": 0.2,
  "vague": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "defect_severity": 0.7234,
  "defect_severity_label": "high"
}
```

#### 4️⃣ Correctness (Fuzzy)
Calculate correctness based on preservation and change accuracy.

**Endpoint:** `POST /api/fuzzy/correctness`

**Request:**
```json
{
  "preservation_correctness": 0.8,
  "change_correctness": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "correctness": 0.7856,
  "correctness_label": "high"
}
```

#### 5️⃣ Requirement Quality (Fuzzy)
Calculate overall requirement quality from defect severity and correctness.

**Endpoint:** `POST /api/fuzzy/requirement-quality`

**Request:**
```json
{
  "defect_severity": 0.3,
  "correctness": 0.8
}
```

**Response:**
```json
{
  "success": true,
  "requirement_quality": 0.8234,
  "requirement_quality_label": "high"
}
```

## 🧪 Testing

```bash
# Test ML models
python test_simple.py

# Test fuzzy systems
python test_fuzzy.py

# Or use curl
curl http://localhost:8001/health
```

## 📁 Project Structure

```
ai-service/
├── app_simple.py         # Main FastAPI application (single file)
├── run_simple.py         # Run script
├── test_simple.py        # Test ML models
├── test_fuzzy.py         # Test fuzzy systems
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

## 💡 Notes

- **First request**: Models download on first use (2-5 minutes)
- **Caching**: Models stay in memory after loading
- **Stop service**: Press CTRL+C in the terminal
- **Interactive docs**: Visit http://localhost:8001/docs to test endpoints in browser

## 🔧 Troubleshooting

**Port already in use?**
```bash
# Edit run_simple.py and change port=8001 to another port
```

**Connection refused?**
```bash
# Make sure service is running:
python run_simple.py
```

**Module not found?**
```bash
# Activate venv and reinstall:
venv\Scripts\activate
pip install -r requirements.txt
```

