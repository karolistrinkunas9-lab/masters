# Backend Integration with Python AI Service

## Overview

The backend now calls the Python AI service to detect quality issues in requirements using the SetFit multi-label classification model.

## Setup

### 1. Environment Variables

Add to your `backend/.env`:

```env
AI_SERVICE_URL=http://localhost:8001
```

### 2. Start Services

```bash
# Terminal 1: Start Python AI Service
cd ai-service
venv\Scripts\activate
python run_simple.py

# Terminal 2: Start Backend
cd backend
npm run start:dev
```

## How It Works

### Issue Detection Flow

1. **Frontend** calls backend to analyze a requirement
2. **Backend** calls Python AI service: `POST /api/inference/requirement-quality`
3. **AI Service** uses SetFit model to detect issues
4. **Backend** returns detected issues to frontend

### Detected Issues

The SetFit model can detect these quality issues:

- **ambiguous**: Requirement contains vague or unclear terms
- **incomplete**: Requirement is missing essential information
- **inconsistent**: Requirement conflicts with other requirements
- **non-verifiable**: Requirement cannot be tested or verified
- **complex**: Requirement is too complex or compound

## Code Changes

### Updated Service: `huggingface.service.ts`

```typescript
async detectIssues(requirementText: string): Promise<string[]> {
  try {
    const { data } = await firstValueFrom(
      this.httpService.post(
        `${this.AI_SERVICE_URL}/api/inference/requirement-quality`,
        { texts: [requirementText] }
      )
    );

    if (data.success && data.predictions?.length > 0) {
      return data.predictions[0].predicted_labels || [];
    }
    return [];
  } catch (error) {
    console.error('Error detecting issues:', error.message);
    return [];
  }
}
```

### Integration in `analyzeRequirement`

The `analyzeRequirement` method now:
1. Performs the main analysis (as before)
2. Calls `detectIssues()` to get quality issues from AI service
3. Merges detected issues into the response

```typescript
// Detect issues using the Python AI service
let detectedIssues: string[] = [];
try {
  detectedIssues = await this.detectIssues(modified);
  console.log('Detected issues from AI service:', detectedIssues);
} catch (error) {
  console.error('Failed to detect issues:', error);
}

// Merge with parsed analysis
if (parsedAnalysis) {
  parsedAnalysis.detected_issues = detectedIssues.length > 0 
    ? detectedIssues 
    : parsedAnalysis.detected_issues;
}
```

## Testing

### Test Issue Detection Directly

```bash
# From backend directory
curl http://localhost:8001/api/inference/requirement-quality \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["The system shall be fast and user-friendly."]
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "predictions": [{
    "text": "The system shall be fast and user-friendly.",
    "predicted_labels": ["ambiguous", "non-verifiable"],
    "probabilities": {
      "ambiguous": 0.87,
      "non-verifiable": 0.92
    }
  }]
}
```

### Test via Backend API

Use your frontend or API client to analyze a requirement. The response will now include `detected_issues` from the AI service.

## Error Handling

The integration includes graceful fallbacks:

- If AI service is unavailable, `detected_issues` will be empty array `[]`
- The main analysis continues even if issue detection fails
- Errors are logged but don't break the analysis flow

## Frontend Display

The frontend should display detected issues as badges or alerts:

```tsx
{analysis.detected_issues?.map(issue => (
  <Badge key={issue} variant="warning">
    {issue}
  </Badge>
))}
```

## Benefits

✅ **Accurate Detection**: Uses trained ML model instead of rule-based detection  
✅ **Consistent**: Same model across all requirements  
✅ **Fast**: Model stays loaded in memory after first use  
✅ **Scalable**: Python service can handle multiple concurrent requests  
✅ **Maintainable**: Single source of truth for issue detection  

## Troubleshooting

### "Error detecting issues from AI service"

**Cause**: Python AI service is not running or not accessible

**Solution**:
```bash
cd ai-service
venv\Scripts\activate
python run_simple.py
```

### Empty detected_issues array

**Cause**: 
- Requirement has no quality issues (good!)
- Or AI service is not responding

**Solution**: Check console logs for error messages

### Connection refused

**Cause**: Wrong `AI_SERVICE_URL` in `.env`

**Solution**: Verify the URL matches where Python service is running:
```env
AI_SERVICE_URL=http://localhost:8001
```

