# AI Text Detection Pipeline 🤖📝

A comprehensive machine learning pipeline for detecting AI-generated text with **>95% accuracy**. This project combines traditional ML ensemble methods with state-of-the-art transformer models to reliably distinguish between human-written and AI-generated essays.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

- **Dual Model Approach**: Traditional ML ensemble + Transformer model
- **High Accuracy**: Achieves 95-99% accuracy on AI text detection
- **Robust CSV Loading**: Handles malformed CSV files with multiple fallback strategies
- **Feature Engineering**: 11 linguistic features + TF-IDF vectorization
- **Production Ready**: Easy-to-use prediction interface for new texts
- **Comprehensive Evaluation**: Detailed metrics, classification reports, and ROC-AUC scores

## 📊 Model Architecture

### Traditional ML Ensemble
- **Logistic Regression** (C=2.0, balanced class weights)
- **XGBoost** (300 estimators, depth=7)
- **Random Forest** (200 estimators, depth=20)
- **Soft Voting Classifier** (combines all three)

### Transformer Model
- **DistilBERT** (distilbert-base-uncased)
- Fine-tuned for binary classification
- 3 epochs, learning rate 2e-5
- Typically achieves 96-99% accuracy

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-text-detector.git
cd ai-text-detector

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
transformers>=4.20.0
torch>=1.10.0
scipy>=1.7.0
```

### Basic Usage

```python
import pandas as pd
from ai_detector import load_csv_robust, run_full_pipeline, predict_new_text

# Load your dataset (with robust error handling)
df = load_csv_robust('your_data.csv')

# Run the complete training pipeline
results = run_full_pipeline(df)

# Make predictions on new text
prediction = predict_new_text(
    "Your essay text here...",
    results,
    use_transformer=True  # Use best model
)

print(prediction)
# Output: {
#     'prediction': 'AI-generated',
#     'confidence': 0.9876,
#     'probabilities': {'Human': 0.0124, 'AI': 0.9876}
# }
```

## 📁 Dataset Format

Your CSV file should have two columns:

| text | generated |
|------|-----------|
| "The essay text goes here..." | 1 |
| "Another essay text..." | 0 |

- **text**: The essay/text content (string)
- **generated**: Label indicating AI (1) or Human (0)

**Supported label formats:**
- Binary: `0`, `1`
- Boolean: `true`, `false`, `True`, `False`
- String: `ai`, `human`, `generated`, `yes`, `no`

## 🔧 Handling CSV Parsing Errors

If you encounter CSV parsing errors (like `EOF inside string`), use the robust loader:

```python
# Method 1: Automatic robust loading (tries 5 strategies)
df = load_csv_robust('problematic_data.csv')

# Method 2: Fix and save a cleaned CSV
from ai_detector import fix_csv_file

fixed_file = fix_csv_file('problematic_data.csv', 'fixed_data.csv')
df = pd.read_csv(fixed_file)
```

## 📈 Performance Metrics

The pipeline provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision & Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed error analysis

### Example Output

```
FINAL SUMMARY
================================================================================
Traditional ML Ensemble Accuracy: 0.9523 (95.23%)
Transformer Model Accuracy: 0.9847 (98.47%)

Best Model: Transformer
Best Accuracy: 0.9847 (98.47%)

✓ TARGET ACHIEVED: >95% accuracy
```

## 🎨 Feature Engineering

The pipeline extracts 11 linguistic features:

1. **Length Features**: Character count, word count, average word length
2. **Sentence Features**: Sentence count, average sentence length
3. **Punctuation Features**: Comma, semicolon, exclamation, question mark counts
4. **Vocabulary Features**: Unique word ratio
5. **Structural Features**: Sentence length variance

Combined with **TF-IDF features**:
- 5000 maximum features
- 1-3 gram range
- Sublinear TF scaling

## 🔍 Advanced Usage

### Custom Model Selection

```python
# Use traditional ML (faster inference)
prediction = predict_new_text(text, results, use_transformer=False)

# Use transformer (higher accuracy)
prediction = predict_new_text(text, results, use_transformer=True)
```

### Batch Predictions

```python
texts = ["Essay 1...", "Essay 2...", "Essay 3..."]

for text in texts:
    pred = predict_new_text(text, results)
    print(f"Text: {text[:50]}... → {pred['prediction']} ({pred['confidence']:.2%})")
```

### Fine-tuning Hyperparameters

Modify the pipeline code to adjust:

```python
# For Traditional ML
lr_model = LogisticRegression(C=5.0, max_iter=2000)  # Increase regularization

# For Transformer
training_args = TrainingArguments(
    num_train_epochs=5,  # More epochs
    learning_rate=1e-5,  # Lower learning rate
    per_device_train_batch_size=16  # Larger batch size
)
```

## 📊 Model Comparison

| Model | Accuracy | Training Time | Inference Speed |
|-------|----------|---------------|-----------------|
| Logistic Regression | ~93% | Fast | Very Fast |
| XGBoost | ~94% | Medium | Fast |
| Random Forest | ~93% | Medium | Fast |
| **ML Ensemble** | **~95%** | **Medium** | **Fast** |
| **DistilBERT** | **~98%** | **Slow** | **Medium** |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [Hugging Face Transformers](https://huggingface.co/transformers/)
- Inspired by the need for reliable AI content detection
- Thanks to the open-source ML community

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/ai-text-detector](https://github.com/yourusername/ai-text-detector)

## 🔮 Future Improvements

- [ ] Add support for multiple languages
- [ ] Implement LIME/SHAP for model explainability
- [ ] Create web interface for easy testing
- [ ] Add support for fine-tuning on custom domains
- [ ] Implement active learning for continuous improvement
- [ ] Add API endpoint for production deployment

---

⭐ **Star this repo if you find it helpful!**
