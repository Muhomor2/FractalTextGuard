# FractalTextGuard

**AI Text Detection via Long-Range Dependence Analysis**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-Non--Commercial-orange.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![No Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)](requirements.txt)

## 🎯 Overview

FractalTextGuard detects AI-generated text by analyzing **fractal patterns** in writing structure. It uses **Detrended Fluctuation Analysis (DFA)** to calculate the Hurst exponent — a mathematical signature that is practically impossible to fake without losing semantic meaning.

> **Key insight**: Human eyes can be fooled by "smart" or "cozy" AI writing styles, but the mathematical structure of Long-Range Dependence (LRD) cannot be faked.

### How It Works

| Property | Human Text | AI-Generated |
|----------|-----------|--------------|
| Hurst exponent (H) | H ≈ 0.5 (random-like) | H > 0.65 (persistent) |
| Compression ratio | Higher (varied) | Lower (repetitive) |
| N-gram patterns | Diverse | Repetitive phrases |
| Entropy | Higher | Lower |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/IgorChechelnitsky/FractalTextGuard.git
cd FractalTextGuard
# No dependencies needed - Python stdlib only!
```

### Basic Usage

```bash
# Check a single file
python analyze.py --file essay.txt

# Check all papers in a folder
python analyze.py --folder submissions/ --output results.json

# Detailed analysis with metrics
python analyze.py --file paper.txt --detailed
```

### Results

| Verdict | Icon | Meaning |
|---------|------|---------|
| **AUTHENTIC** | ✅ | Text appears human-written |
| **SUSPICIOUS** | ⚠️ | Some AI patterns detected, review recommended |
| **AI_DETECTED** | 🤖 | Strong AI generation indicators |

---

## 📚 Use Cases

### 🎓 Academic (EdTech)

**Dissertation & Essay Verification**

Unlike plagiarism checkers that search for matches, FractalTextGuard sees the "genetic code" of text structure.

```bash
python analyze.py --folder student_papers/ --output academic_report.json
```

**Scientific Article Review**

Detect "artificial smoothness" — when text is too perfect, too structured.

---

### 🔒 Cybersecurity

**Bot Detection in Social Media**

Mass-generated propaganda leaves traces in anomalous Hurst exponents.

**Phishing Email Analysis**

AI-generated phishing emails show characteristic repetitive patterns.

```bash
python analyze.py --folder suspicious_emails/ --detailed
```

---

### ⚖️ Legal & Compliance

**Document Authenticity**

Verify if witness statements or legal documents were AI-generated.

**Code Audit**

Detect automatically generated code sections that may contain vulnerabilities.

---

### 💰 Financial

**News Verification**

Distinguish real market news from AI-generated manipulation.

**Report Analysis**

Detect signs of automated report generation.

---

## 📊 Output Examples

### Quick Check
```
============================================================
File: student_essay.txt
============================================================

  Result: ✅ AUTHENTIC
  Confidence: 80%

============================================================
```

### Detailed Analysis
```
============================================================
File: suspicious_document.txt
============================================================

  Result: 🤖 AI_DETECTED
  Confidence: 92%

  ⚠️  Warnings:
      - Very high Hurst exponent: 0.782
      - High compressibility: 0.312
      - High phrase repetition: 28.3%

  📊 Detailed Metrics:
      Hurst exponent (H): 0.782
      Compression ratio: 0.312
      Repetition rate: 15.2%

  📝 Interpretation:
      H: Highly persistent (AI indicator)
      compression: High repetition
============================================================
```

### Batch Processing
```
📁 Analyzed 25 files from submissions/

Summary:
  ✅ AUTHENTIC:   20
  ⚠️  SUSPICIOUS:  3
  🤖 AI_DETECTED: 2

⚠️  Files requiring attention:
    🤖 paper_17.txt: AI_DETECTED
    🤖 paper_23.txt: AI_DETECTED
    ⚠️  paper_05.txt: SUSPICIOUS
```

---

## 🔬 Scientific Background

### The Hurst Exponent

The tool calculates **H** via Detrended Fluctuation Analysis:

| H value | Interpretation |
|---------|----------------|
| H < 0.5 | Anti-persistent (unusual) |
| H ≈ 0.5 | Random (typical human writing) |
| H > 0.65 | Suspicious (possible AI) |
| H > 0.75 | Strong AI indicator |

### Why LRD Cannot Be Faked

Language models optimize for local coherence but cannot replicate the **long-range fractal structure** of human cognition. This signature is:

- **Computationally expensive** to simulate
- **Semantically destructive** if artificially introduced
- **Statistically detectable** across multiple scales

---

## 📁 Package Structure

```
FractalTextGuard/
├── analyze.py              # Main entry point
├── README.md
├── LICENSE                 # Non-commercial
├── CITATION.cff
├── config.json
├── src/
│   ├── analyzer_core.py    # Core analysis
│   └── gsl_lrd_v25.py      # Full analyzer
├── examples/
│   ├── academic_essay.txt
│   ├── phishing_email.txt
│   ├── code_sample.py
│   └── sample_*.txt
├── tests/
│   └── test_analyzer.py
└── stress_test_*.py        # Industrial testing
```

---

## 📜 License

**Non-Commercial Use License**

| Use | Allowed |
|-----|---------|
| Personal | ✅ Free |
| Academic | ✅ Free |
| Educational | ✅ Free |
| Research | ✅ Free |
| **Commercial** | ❌ Requires written permission |

See [LICENSE](LICENSE) for details.

---

## 📖 Citation

```bibtex
@software{chechelnitsky2026fractaltextguard,
  author = {Chechelnitsky, Igor},
  title = {FractalTextGuard: AI Text Detection via Long-Range Dependence Analysis},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/IgorChechelnitsky/FractalTextGuard}
}
```

---

## 👤 Author

**Igor Chechelnitsky**  
ORCID: [0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)
Contact : Facebook.com (only)

---

*Fractal analysis reveals what human eyes cannot see.*
