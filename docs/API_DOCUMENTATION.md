# TEXT-AUTH API Documentation

## Overview

The TEXT-AUTH API provides evidence-based text forensics and statistical consistency assessment through a RESTful interface. This document covers all endpoints, request/response formats, authentication, rate limiting, and integration examples.

**API Version:** 1.0.0  

---

## Table of Contents

1. [Authentication & Security](#authentication--security)
2. [Rate Limiting](#rate-limiting)
3. [Common Response Format](#common-response-format)
4. [Error Handling](#error-handling)
5. [Core Endpoints](#core-endpoints)
   - [Text Analysis](#text-analysis)
   - [File Analysis](#file-analysis)
   - [Batch Analysis](#batch-analysis)
6. [Report Endpoints](#report-endpoints)
7. [Utility Endpoints](#utility-endpoints)
8. [Best Practices](#best-practices)

---

## Authentication & Security

### API Key Authentication

*Authentication is not enforced in the current deployment. API key authentication may be added in future versions.*


## Rate Limiting

*Rate limiting is not enforced at the application level. Deployments should use an external gateway (NGINX, API Gateway, Cloudflare) to enforce rate limits if required.*

---
## Common Response Format

All successful responses follow this structure:

```json
{
  "status": "success",
  "analysis_id": "...",
  "detection_result": {...},
  "highlighted_html": "...",
  "reasoning": {...},
  "processing_time": 2.34,
  "timestamp": "..."
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "error": "Invalid domain...",
  "timestamp": "..."
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `TEXT_TOO_LONG` | Text exceeds maximum length (50,000 chars) | Split into multiple requests |
| `FILE_TOO_LARGE` | File exceeds size limit | Compress or split file |
| `UNSUPPORTED_FORMAT` | File format not supported | Use .txt, .pdf, .docx, .doc, or .md |
| `EXTRACTION_FAILED` | Document text extraction failed | Ensure file is not corrupted or password-protected |
| `MODEL_UNAVAILABLE` | Required model temporarily unavailable | Retry after a few minutes |

---

## Core Endpoints

### Text Analysis

**Endpoint:** `POST /api/analyze`

Analyze raw text for statistical consistency patterns and forensic signals.

#### Request

**Headers:**
```http
Content-Type: application/json
```

**Body:**
```json
{
  "text": "Your text content here...",
  "domain": "academic",
  "enable_highlighting": true,
  "skip_expensive_metrics": false,
  "use_sentence_level": true,
  "include_metrics_summary": true,
  "generate_report": false
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | **Yes** | - | Text to analyze (50-50,000 chars) |
| `domain` | string | No | `null` (auto-detect) | Content domain (see [Domains](#supported-domains)) |
| `enable_highlighting` | boolean | No | `true` | Generate sentence-level highlights |
| `skip_expensive_metrics` | boolean | No | `false` | Skip computationally expensive metrics for faster results |
| `use_sentence_level` | boolean | No | `true` | Use sentence-level granularity for highlighting |
| `include_metrics_summary` | boolean | No | `true` | Include metric summaries in highlights |
| `generate_report` | boolean | No | `false` | Generate downloadable PDF/JSON report |

#### Response

```json
{
  "status": "success",
  "analysis_id": "analysis_1735555800000",
  "detection_result": {
    "ensemble_result": {
      "final_verdict": "Synthetic",
      "overall_confidence": 0.89,
      "synthetic_probability": 0.92,
      "authentic_probability": 0.08,
      "uncertainty_score": 0.23,
      "decision_boundary_distance": 0.42
    },
    "metric_results": {
      "perplexity": {
        "synthetic_probability": 0.94,
        "confidence": 0.91,
        "raw_score": 15.23,
        "evidence_strength": "strong"
      },
      "entropy": {
        "synthetic_probability": 0.88,
        "confidence": 0.85,
        "raw_score": 4.67,
        "evidence_strength": "moderate"
      },
      "structural": {
        "synthetic_probability": 0.91,
        "confidence": 0.87,
        "burstiness": -0.12,
        "uniformity": 0.85,
        "evidence_strength": "strong"
      },
      "linguistic": {
        "synthetic_probability": 0.86,
        "confidence": 0.82,
        "pos_diversity": 0.42,
        "mean_tree_depth": 4.2,
        "evidence_strength": "moderate"
      },
      "semantic": {
        "synthetic_probability": 0.93,
        "confidence": 0.88,
        "coherence_mean": 0.91,
        "coherence_variance": 0.03,
        "evidence_strength": "strong"
      },
      "multi_perturbation_stability": {
        "synthetic_probability": 0.89,
        "confidence": 0.84,
        "stability_score": 0.12,
        "evidence_strength": "moderate"
      }
    },
    "domain_prediction": {
      "primary_domain": "academic",
      "confidence": 0.94,
      "alternative_domains": [
        {"domain": "technical_doc", "probability": 0.23},
        {"domain": "science", "probability": 0.18}
      ]
    },
    "processed_text": {
      "word_count": 487,
      "sentence_count": 23,
      "paragraph_count": 5,
      "avg_sentence_length": 21.2,
      "language": "en"
    }
  },
  "highlighted_html": "<div class=\"text-forensics-highlight\">...</div>",
  "reasoning": {
    "summary": "The text exhibits strong statistical consistency patterns typical of language model generation...",
    "key_indicators": [
      "Unusually uniform sentence structure (burstiness: -0.12)",
      "High semantic coherence across all sentences (mean: 0.91)",
      "Low perplexity variance indicating predictable token sequences"
    ],
    "confidence_factors": {
      "supporting_evidence": [
        "6/6 metrics indicate synthetic patterns",
        "Strong cross-metric agreement (correlation: 0.87)"
      ],
      "uncertainty_sources": [
        "Domain-specific terminology may affect baseline expectations"
      ]
    },
    "metric_contributions": {
      "perplexity": 0.28,
      "entropy": 0.19,
      "structural": 0.16,
      "semantic": 0.17,
      "linguistic": 0.12,
      "multi_perturbation_stability": 0.08
    }
  },
  "report_files": null,
  "processing_time": 2.34,
  "timestamp": "2025-12-30T10:30:00Z"
}
```

#### Verdict Interpretation

| Verdict | Probability Range | Interpretation |
|---------|-------------------|----------------|
| **Synthetic** | > 0.70 | High consistency with language model generation patterns |
| **Likely Synthetic** | 0.55 - 0.70 | Moderate consistency with synthetic patterns |
| **Inconclusive** | 0.45 - 0.55 | Insufficient evidence for confident assessment |
| **Likely Authentic** | 0.30 - 0.45 | Moderate consistency with human authorship patterns |
| **Authentic** | < 0.30 | High consistency with human authorship patterns |

**Important:** These verdicts represent statistical consistency assessments, not definitive authorship claims.

#### Highlighting Color Key

| Color | Meaning | Probability Range |
|-------|---------|-------------------|
| 🔴 Red | Strong synthetic signals | > 0.80 |
| 🟠 Orange | Moderate synthetic signals | 0.60 - 0.80 |
| 🟡 Yellow | Weak signals | 0.40 - 0.60 |
| 🟢 Green | Authentic signals | < 0.40 |

---

### File Analysis

**Endpoint:** `POST /api/analyze/file`

Analyze uploaded documents (PDF, DOCX, DOC, TXT, MD).

#### Request

**Headers:**
```http
Content-Type: multipart/form-data
```

**Body (form-data):**
```
file: [binary file data]
domain: "academic"
skip_expensive_metrics: false
use_sentence_level: true
include_metrics_summary: true
generate_report: false
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | **Yes** | - | Document file (max 25MB) |
| `domain` | string | No | `null` | Content domain override |
| `skip_expensive_metrics` | boolean | No | `false` | Skip expensive metrics |
| `use_sentence_level` | boolean | No | `true` | Sentence-level highlighting |
| `include_metrics_summary` | boolean | No | `true` | Include metric summaries |
| `generate_report` | boolean | No | `false` | Generate report |

#### Supported File Formats

| Format | Extensions | Max Size | Notes |
|--------|-----------|----------|-------|
| Plain Text | .txt, .md | 25MB | UTF-8 encoding recommended |
| PDF | .pdf | 25MB | Text-based PDFs; OCR not supported |
| Word | .docx, .doc | 25MB | Modern and legacy formats |

#### Response

Same structure as [Text Analysis](#text-analysis) with additional `file_info`:

```json
{
  "status": "success",
  "analysis_id": "file_1735555800000",
  "file_info": {
    "filename": "research_paper.pdf",
    "file_type": ".pdf",
    "pages": 12,
    "extraction_method": "pdfplumber",
    "highlighted_html": true
  },
  "detection_result": { /* same as text analysis */ },
  "highlighted_html": "...",
  "reasoning": { /* same as text analysis */ },
  "processing_time": 4.12,
  "timestamp": "2025-12-30T10:30:00Z"
}
```

#### cURL Example

```bash
curl -X POST https://your-domain.com/api/analyze/file \
  -F "file=@/path/to/document.pdf" \
  -F "domain=academic" \
  -F "generate_report=true"
```

---

### Batch Analysis

**Endpoint:** `POST /api/analyze/batch`

Analyze multiple texts in a single request for efficiency.

#### Request

```json
{
  "texts": [
    "First text to analyze...",
    "Second text to analyze...",
    "Third text to analyze..."
  ],
  "domain": "academic",
  "skip_expensive_metrics": true,
  "generate_reports": false
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `texts` | array[string] | **Yes** | - | 1-100 texts to analyze |
| `domain` | string | No | `null` | Apply same domain to all texts |
| `skip_expensive_metrics` | boolean | No | `true` | Skip expensive metrics (recommended for batch) |
| `generate_reports` | boolean | No | `false` | Generate reports for each text |

#### Response

```json
{
  "status": "success",
  "batch_id": "batch_1735555800000",
  "total": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "index": 0,
      "status": "success",
      "detection": {
        "ensemble_result": { /* ... */ },
        "metric_results": { /* ... */ }
      },
      "reasoning": { /* ... */ },
      "report_files": null
    },
    {
      "index": 1,
      "status": "success",
      "detection": { /* ... */ }
    },
    {
      "index": 2,
      "status": "error",
      "error": "Text too short (minimum 50 characters)"
    }
  ],
  "processing_time": 8.92,
  "timestamp": "2025-12-30T10:30:00Z"
}
```

#### Performance Tips

- Set `skip_expensive_metrics: true` for faster batch processing
- Keep batch size under 50 texts for optimal performance
- Consider parallel API calls for batches > 100 texts
- Monitor `processing_time` to adjust batch sizes

---

## Report Endpoints

### Generate Report

**Endpoint:** `POST /api/report/generate`

Generate detailed PDF/JSON reports for cached analyses.

#### Request

**Headers:**
```http
Content-Type: application/x-www-form-urlencoded
```

**Body:**
```
analysis_id=analysis_1735555800000
formats=json,pdf
include_highlights=true
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `analysis_id` | string | **Yes** | - | Analysis ID from previous request |
| `formats` | string | No | `"json,pdf"` | Comma-separated formats |
| `include_highlights` | boolean | No | `true` | Include sentence highlights in report |

#### Response

```json
{
  "status": "success",
  "analysis_id": "analysis_1735555800000",
  "reports": {
    "json": "analysis_1735555800000.json",
    "pdf": "analysis_1735555800000.pdf"
  },
  "timestamp": "2025-12-30T10:30:00Z"
}
```

### Download Report

**Endpoint:** `GET /api/report/download/{filename}`

Download a generated report file.

#### Request

```http
GET /api/report/download/analysis_1735555800000.pdf
```

#### Response

Binary file download with appropriate `Content-Type` header.

**Headers:**
```http
Content-Type: application/pdf
Content-Disposition: attachment; filename="analysis_1735555800000.pdf"
Content-Length: 524288
```

---

## Utility Endpoints

### Health Check

**Endpoint:** `GET /health`

Check API health and model availability.

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 86400.5,
  "models_loaded": {
    "orchestrator": true,
    "highlighter": true,
    "reporter": true,
    "reasoning_generator": true,
    "document_extractor": true,
    "analysis_cache": true,
    "parallel_executor": true
  }
}
```

### List Domains

**Endpoint:** `GET /api/domains`

Get all supported content domains with descriptions.

#### Response

```json
{
  "domains": [
    {
      "value": "general",
      "name": "General",
      "description": "General-purpose text without domain-specific structure"
    },
    {
      "value": "academic",
      "name": "Academic",
      "description": "Academic papers, essays, research"
    },
    {
      "value": "creative",
      "name": "Creative",
      "description": "Creative writing, fiction, poetry"
    },
    {
      "value": "technical_doc",
      "name": "Technical Doc",
      "description": "Technical documentation, manuals, specs"
    }
    // ... 12 more domains
  ]
}
```

### Supported Domains

| Domain | Use Cases | Threshold Adjustments |
|--------|-----------|----------------------|
| `general` | Default fallback | Balanced weights |
| `academic` | Research papers, essays | Higher linguistic weight |
| `creative` | Fiction, poetry | Higher entropy/structural |
| `ai_ml` | ML papers, technical AI content | Semantic prioritized |
| `software_dev` | Code docs, READMEs | Structural relaxed |
| `technical_doc` | Manuals, specs | Higher semantic weight |
| `engineering` | Technical reports | Balanced technical focus |
| `science` | Scientific papers | Academic-like calibration |
| `business` | Reports, proposals | Formal structure emphasis |
| `legal` | Contracts, court filings | Strict structural patterns |
| `medical` | Clinical notes, research | Domain-specific terminology |
| `journalism` | News articles | Balanced, lower burstiness |
| `marketing` | Ad copy, campaigns | Creative elements |
| `social_media` | Posts, casual writing | Relaxed metrics, high perplexity weight |
| `blog_personal` | Personal blogs, diaries | Creative + casual mix |
| `tutorial` | How-to guides | Instructional patterns |

### Cache Statistics

**Endpoint:** `GET /api/cache/stats`

Get analysis cache statistics (admin only).

#### Response

```json
{
  "cache_size": 42,
  "max_size": 100,
  "ttl_seconds": 3600
}
```

### Clear Cache

**Endpoint:** `POST /api/cache/clear`

Clear analysis cache (admin only).

#### Response

```json
{
  "status": "success",
  "message": "Cache cleared"
}
```

---

## Best Practices

### Optimization Tips

1. **Domain Selection**
   - Always specify domain when known for better accuracy
   - Use `/api/domains` to explore available options
   - Let system auto-detect only when domain is truly unknown

2. **Performance**
   - Set `skip_expensive_metrics: true` for faster results when speed matters
   - Use batch API for multiple texts instead of sequential single requests
   - Cache `analysis_id` to regenerate reports without reanalysis

3. **Accuracy**
   - Provide clean, well-formatted text (remove excessive whitespace)
   - Minimum 100 words recommended for reliable results
   - Avoid mixing languages in single analysis

4. **Rate Limiting**
   - Implement exponential backoff on 429 responses
   - Monitor `X-RateLimit-Remaining` header
   - Upgrade tier if consistently hitting limits

5. **Error Handling**
   - Always check `status` field in response
   - Log `request_id` for support requests
   - Implement retry logic with jitter for transient errors

### Security Recommendations

1. **API Key Management**
   - Rotate keys every 90 days
   - Use separate keys for dev/staging/production
   - Revoke compromised keys immediately

2. **Data Privacy**
   - Never send PII unless absolutely necessary
   - Use client-side redaction before API calls
   - Enable data retention policies in dashboard

3. **Input Validation**
   - Sanitize user input before sending to API
   - Validate file types client-side
   - Implement size limits before upload

---

## Version History:

- **1.0.0** (2025-12-30): Initial release
  - 6 forensic metrics
  - 16 domain support
  - PDF/JSON reporting
  - Batch processing

---

## Appendix

### Complete Domain List with Aliases

```python
DOMAIN_ALIASES = {
    'general': ['default', 'generic'],
    'academic': ['education', 'research', 'scholarly', 'university'],
    'creative': ['fiction', 'literature', 'story', 'narrative'],
    'ai_ml': ['ai', 'ml', 'machinelearning', 'neural'],
    'software_dev': ['software', 'code', 'programming', 'dev'],
    'technical_doc': ['technical', 'tech', 'documentation', 'manual'],
    'engineering': ['engineer'],
    'science': ['scientific'],
    'business': ['corporate', 'commercial', 'enterprise'],
    'legal': ['law', 'contract', 'court'],
    'medical': ['healthcare', 'clinical', 'medicine', 'health'],
    'journalism': ['news', 'reporting', 'media', 'press'],
    'marketing': ['advertising', 'promotional', 'brand', 'sales'],
    'social_media': ['social', 'casual', 'informal', 'posts'],
    'blog_personal': ['blog', 'personal', 'diary', 'lifestyle'],
    'tutorial': ['guide', 'howto', 'instructional', 'walkthrough']
}
```

### Metric Weight Defaults

```python
DEFAULT_WEIGHTS = {
    'perplexity': 0.25,
    'entropy': 0.20,
    'structural': 0.15,
    'semantic': 0.15,
    'linguistic': 0.15,
    'multi_perturbation_stability': 0.10
}
```

### Response Time Estimates

| Operation | Min | Avg | Max | P95 |
|-----------|-----|-----|-----|-----|
| Text Analysis (500 words) | 1.2s | 2.3s | 4.5s | 3.8s |
| File Analysis (PDF, 10 pages) | 2.5s | 4.1s | 8.2s | 6.9s |
| Batch (10 texts) | 5.8s | 9.2s | 15.3s | 13.1s |
| Report Generation | 0.3s | 0.8s | 2.1s | 1.5s |

---

*Last Updated: December 30, 2025*  
*API Version: 1.0.0*  
*Documentation Version: 1.0.0*