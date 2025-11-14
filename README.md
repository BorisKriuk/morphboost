# 🚀 MorphBoost: Self-Organizing Universal Gradient Boosting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)

**MorphBoost** is a novel gradient boosting algorithm that adapts its internal architecture during training through self-morphing tree structures. It outperforms XGBoost by 0.84% on average across diverse datasets while maintaining scikit-learn compatibility.

---

## 🏆 Performance Highlights

| Rank | Model | Avg Accuracy | Wins | Top-3 Finishes |
|------|-------|--------------|------|----------------|
| 🥇 | **MorphBoost** | **0.9009** | **4** | **6** |
| 🥈 | GradBoost | 0.8959 | 2 | 4 |
| 🥉 | HistGradBoost | 0.8952 | 1 | 6 |
| 4 | XGBoost | 0.8934 | 0 | 1 |

### Key Achievements

✅ **Overall Winner**: MorphBoost (0.9009 accuracy)

✅ **Outperforms XGBoost**: +0.84% accuracy improvement

✅ **Most Consistent**: σ=0.0948 (lowest variance)

✅ **Most Robust**: 0.6650 minimum accuracy (highest floor)

✅ **Dataset Wins**: 4/10 (40% win rate)

✅ **Top-3 Finishes**: 6/30 possible (20%)

---

## ✨ Key Features

- 🧬 **Self-Morphing Architecture** - Trees adapt their split functions during training based on data complexity
- 🎯 **Adaptive Learning** - Dynamic learning rate scheduling with problem-aware adjustments
- 🔄 **Interaction Detection** - Automatic feature interaction discovery and exploitation
- ⚡ **Fast Mode** - Optimized vectorized operations for 10x+ speedup
- 🎨 **Multi-Class Support** - Native support for binary, multiclass, and regression tasks
- 📊 **Feature Importance** - Enhanced importance calculation with morphing scores
- 🧠 **Neural Embeddings** - Optional neural-inspired feature transformations
- ⚙️ **Scikit-learn Compatible** - Drop-in replacement for sklearn estimators

---

## 📦 Installation

bash
git clone https://github.com/BorisKriuk/morphboost.git
cd morphboost
pip install -r requirements.txt


## 🚀 Quick Start

python
from morphboost import MorphBoost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MorphBoost(n_estimators=100, learning_rate=0.3, fast_mode=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


## 🧪 Running Benchmarks

bash
python test_run.py


## 📝 License

MIT License - see LICENSE file for details
