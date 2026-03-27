# ⚡ Electric Load Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()
[![Flask](https://img.shields.io/badge/Flask-Web-orange?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-scikit--learn-red?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)

A robust, production-ready data mining and machine learning solution for high-accuracy electricity load forecasting. This system leverages advanced clustering algorithms and ensemble machine learning techniques to analyze hourly electricity demand patterns across major U.S. cities, delivering actionable insights and precise 24-hour load predictions through an intuitive web-based analytics dashboard.

**Key Capabilities:**
- 🎯 Multi-algorithm clustering analysis (K-Means, DBSCAN, Hierarchical)
- 📈 Ensemble forecasting with 4+ predictive models (XGBoost, Random Forest, Linear Regression)
- 📊 Real-time interactive visualizations with PCA & t-SNE dimensionality reduction
- 🔍 Anomaly detection and pattern discovery
- 🌐 RESTful API with Flask web interface

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

The **Electric Load Forecasting System** represents a comprehensive solution for predictive analytics in the energy sector. Built with enterprise-grade Python libraries, this system employs three interconnected subsystems:

1. **Clustering Analysis**: Identifies groups of similar consumption–weather patterns across cities and time periods
2. **Predictive Modeling**: Creates forecasts of future electricity demand using multiple models and ensemble techniques
3. **Front-End Interface**: Provides a user-friendly web application for data exploration and visualization

## 📂 Project Structure

```
electric-load-forecasting/
│
├── 📄 app.py                          # Flask application entry point & API routes
├── 📄 requirements.txt                # Production dependencies
├── 📄 requirements_minimal.txt        # Minimal dependency set
├── 📄 requirements_flexible.txt       # Extended dependency options
├── 📄 project_documentation.md        # Comprehensive project documentation
├── 📄 README.md                       # This file
│
├── 📁 data/                           # Raw datasets
│   ├── full_dataset_daily.csv         # Daily aggregated electricity load
│   ├── full_dataset_hourly.csv        # Hourly granularity data (primary)
│   └── full_dataset_weekly.csv        # Weekly aggregated data
│
├── 📁 processed_data/                 # Processed & cleaned datasets
│   └── final_preprocess.py            # Data preprocessing pipeline
│
├── 📁 scripts/                        # Analysis & forecasting scripts
│   ├── run_forecasting.py             # Forecasting pipeline (XGBoost, RF, LR, Ensemble)
│   └── run_clustering.py              # Clustering analysis (K-Means, DBSCAN, Hierarchical)
│
├── 📁 models/                         # Trained ML models (pickled)
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl                     # StandardScaler for feature normalization
│
├── 📁 templates/                      # Jinja2 HTML templates
│   └── index.html                     # Interactive dashboard interface
│
├── 📁 static/                         # Frontend assets
│   ├── custom.css                     # Styling & responsive design
│   └── app.js                         # Client-side functionality
│
├── 📁 visualizations/                 # Generated plots & charts
│   ├── clustering/
│   │   ├── kmeans_clusters_pca.png
│   │   ├── dbscan_clusters_pca.png
│   │   └── hierarchical_dendrogram.png
│   ├── forecasting/
│   │   ├── forecast_*.png             # City-specific forecasts
│   │   └── ensemble*.png              # Ensemble model predictions
│   ├── dimensionality_reduction/
│   │   ├── pca_2d.png
│   │   ├── pca_biplot.png
│   │   └── tsne.png
│   └── performance/
│       ├── model_performance.png
│       ├── feature_importance.png
│       └── error_distribution.png
│
├── 📁 forecasting_results/            # Model predictions & metrics
│   ├── next_day_predictions.csv
│   └── model_performance.csv          # RMSE, MAE, MAPE, R² scores
│
└── 📁 clustering_results/             # Clustering analysis outputs
    ├── kmeans_cluster_means.csv
    ├── anomalies_detected.csv         # Isolated data points
    ├── cluster_descriptions.csv
    ├── silhouette_scores.csv
    ├── cluster_relative_differences.csv
    └── ensemble_weights.npy           # Model combination weights
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│         ELECTRIC LOAD FORECASTING SYSTEM               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────┐ │
│  │  Data Layer    │  │ Processing     │  │ ML/AI    │ │
│  │                │  │ Pipeline       │  │ Engine   │ │
│  │ • CSV Files    │  │                │  │          │ │
│  │ • Time Series  │  │ • Clustering   │  │ • XGBoost│ │
│  │ • Weather Data │  │ • Feature Eng. │  │ • RF     │ │
│  │                │  │ • Scaling      │  │ • LR     │ │
│  └────────────────┘  └────────────────┘  └──────────┘ │
│         ↓                   ↓                    ↓      │
│  ┌──────────────────────────────────────────────────┐  │
│  │         RESTful API (Flask)                      │  │
│  │  /api/forecast, /api/cluster, /api/metrics      │  │
│  └──────────────────────────────────────────────────┘  │
│         ↓                                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │    Interactive Web Dashboard                    │  │
│  │  • Load Visualization                          │  │
│  │  • Clustering Results                          │  │
│  │  • 24-Hour Forecasts                           │  │
│  │  • Model Performance Metrics                   │  │
│  │  • City Comparisons                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 📊 System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python Version** | 3.10 or higher |
| **OS** | Windows, macOS, Linux |
| **RAM** | 4GB minimum (8GB+ recommended) |
| **Storage** | 2GB free space |
| **CPU** | Multi-core processor (4+ cores recommended) |
| **Browser** | Chrome, Firefox, Edge (latest versions) |
| **Package Manager** | Conda or pip |

## 🚀 Installation

### Prerequisites
- Ensure [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed
- Git for cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/electric-load-forecasting.git
cd electric-load-forecasting
```

### Step 2: Create Conda Environment

Create an isolated Conda environment with Python 3.10:

```bash
conda create -n elf python=3.10.20
```

Activate the environment:

```bash
# On Windows
conda activate elf

# On macOS/Linux
source activate elf
```

You should see `(elf)` prefix in your terminal, indicating the environment is active.

### Step 3: Install Dependencies

Install all required packages into the active environment:

```bash
pip install -r requirements.txt
```

**Alternative:** If you prefer minimal dependencies:
```bash
pip install -r requirements_minimal.txt
```

**Flexible setup** (with optional packages):
```bash
pip install -r requirements_flexible.txt
```

### Step 4: Verify Installation

Confirm that all packages are installed correctly:

```bash
python -c "import flask, pandas, numpy, sklearn, xgboost; print('✓ All dependencies installed successfully')"
```

Expected output:
```
✓ All dependencies installed successfully
```

## 🎬 Quick Start

### Running the Web Application

```bash
# Ensure you're in the project root with (elf) environment activated
python app.py
```

Expected console output:
```
 * Running on http://127.0.0.1:5000
 * Cities loaded: 10/10 ✓
 * API endpoints ready
 * Press CTRL+C to quit
```

Navigate to: **http://127.0.0.1:5000** in your web browser

### Running Individual Scripts

```bash
# Forecasting analysis
python scripts/run_forecasting.py

# Clustering analysis
python scripts/run_clustering.py

# Data preprocessing
python processed_data/final_preprocess.py
```

## 🎮 Usage Guide

### Web Dashboard Features

The interactive web dashboard provides comprehensive analytics across six main tabs:

#### 1. **Load Data** 📈
- Real-time and historical electricity consumption visualization
- Temperature overlay for correlation analysis
- Date range filtering and city selection
- Data quality indicators and anomaly highlights

#### 2. **Clustering Analysis** 🎯
- Real-time clustering with three algorithms:
  - **K-Means**: Partition-based clustering with configurable K parameter
  - **DBSCAN**: Density-based anomaly detection
  - **Hierarchical**: Agglomerative clustering with dendrogram visualization
- Interactive PCA 2D/3D projections
- t-SNE dimensionality reduction visualization
- Silhouette score evaluation

#### 3. **Forecasting** 🔮
- 24-hour load predictions with confidence intervals
- Multiple model forecasts (XGBoost, Random Forest, Linear Regression)
- Ensemble model predictions combining all algorithms
- Prediction uncertainty quantification

#### 4. **Model Performance** 📊
- Comparative performance metrics (RMSE, MAE, MAPE, R²)
- Feature importance analysis
- Error distribution plots
- Model accuracy rankings

#### 5. **City Comparison** 🌍
- Multi-city load pattern analysis
- Comparative consumption trends
- Peak demand identification
- Seasonal pattern recognition

#### 6. **Daily Pattern** 📅
- Average daily consumption profiles
- Hourly granularity analysis
- Typical vs. anomalous day comparison
- Demand forecasting for next 24 hours

### Control Panel Options

**Left Sidebar Controls:**
- 🏙️ **City Selection**: Choose from 10 major U.S. cities
- 📅 **Date Range**: Filter data by time period
- 🔧 **Algorithm Selection**: Switch between clustering methods
- 📊 **K-Value Slider**: Adjust number of clusters (2-10)
- 🤖 **Model Selection**: Choose forecasting algorithm

## ✨ Key Features

### Data Processing
- **Robust Data Pipeline**: Automated missing value imputation using multiple strategies
- **Feature Engineering**: Temporal features (hour, day, month, season) + weather correlation analysis
- **Normalization**: StandardScaler for feature scaling and model compatibility
- **Time Series Handling**: Proper lag features and rolling window statistics

### Clustering Capabilities
| Algorithm | Strengths | Use Case |
|-----------|-----------|----------|
| **K-Means** | Fast, interpretable, optimal for spherical clusters | General-purpose pattern discovery |
| **DBSCAN** | Density-aware, anomaly detection, no predetermined K | Identifying outliers and noise |
| **Hierarchical** | Dendrograms, flexible distance metrics | Understanding cluster relationships |

### Forecasting Models
| Model | Accuracy | Speed | Interpretability |
|-------|----------|-------|-----------------|
| **Linear Regression** | Baseline | ⚡⚡⚡ | ★★★★★ |
| **Random Forest** | High | ⚡⚡ | ★★★ |
| **XGBoost** | Very High | ⚡⚡ | ★★ |
| **Ensemble** | Best | ⚡ | ★★★ |

### Advanced Analytics
- **Ensemble Methods**: Weighted combination of multiple models for optimal predictions
- **Anomaly Detection**: Isolation Forest and statistical methods
- **Dimensionality Reduction**: PCA with 95% variance retention + t-SNE
- **Performance Metrics**: RMSE, MAE, MAPE, R², cross-validated scores
- **Confidence Intervals**: Prediction uncertainty quantification

### User Experience
- 🎨 Modern, responsive web interface
- ⚡ Real-time data updates and visualizations
- 🖱️ Intuitive controls with instant feedback
- 📱 Mobile-friendly design
- ♿ Accessibility compliance

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **Port 5000 Already in Use**
```bash
# Find process using port 5000
netstat -ano | findstr :5000  # Windows
lsof -i :5000                  # macOS/Linux

# Kill the process or use different port
set FLASK_PORT=5001
python app.py
```

#### 2. **Module Import Errors**
```bash
# Verify environment is active and packages installed
conda activate elf
pip list | grep flask  # Check specific packages

# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

#### 3. **Missing City Data**
The application automatically generates synthetic data if files are missing. Check console for:
```
[INFO] City X missing from dataframe, generating synthetic data...
```

#### 4. **Slow Predictions on First Run**
- Initial model loading takes 30-60 seconds
- Subsequent API calls are cached for performance
- XGBoost model is larger (~2.3MB) than linear regression

#### 5. **Memory Issues with Large Datasets**
```bash
# Increase available memory or use minimal requirements
pip install -r requirements_minimal.txt

# Or reduce data scope in app.py
MAX_ROWS = 50000  # Limit dataset size
```

#### 6. **Web Interface Not Loading**
```bash
# Clear browser cache (Ctrl+Shift+Del)
# Check Flask console for error messages
# Verify JavaScript is enabled
# Try incognito/private browsing mode
```

### Getting Help

1. **Check logs**: Review full Flask console output
2. **Verify setup**: Run `python -c "import flask; print(flask.__version__)"`
3. **Documentation**: See [project_documentation.md](project_documentation.md)
4. **GitHub Issues**: Submit detailed bug reports with environment info

## 🧪 Testing

Run tests to verify installation:

```bash
# Test data loading
python -c "from scripts.run_forecasting import *; print('✓ Scripts loaded')"

# Test web server
python app.py --test

# Test specific model
python -c "import pickle; pickle.load(open('models/xgboost_model.pkl', 'rb')); print('✓ Models loaded')"
```

## 📊 Performance Benchmarks

| Operation | Time | Resource |
|-----------|------|----------|
| Data Loading | ~2-5s | 150MB RAM |
| Cluster Analysis (K-Means, K=5) | ~1-3s | 200MB RAM |
| Forecasting (24hr, single model) | ~500ms | 300MB RAM |
| Ensemble Prediction | ~1-2s | 450MB RAM |
| Full Dashboard Load | ~8-12s | 600MB RAM |

**System**: Intel i5, 8GB RAM, 256GB SSD

## 🔄 Workflow

### Development Workflow
```bash
# 1. Activate environment
conda activate elf

# 2. Launch development server (with auto-reload)
python app.py

# 3. Open browser to http://localhost:5000

# 4. Make code changes (Flask auto-reloads)

# 5. Commit changes
git add .
git commit -m "feat: add feature description"
```

### Production Deployment
```bash
# Use WSGI server instead of Flask development server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or use Docker (if Dockerfile exists)
docker build -t elf .
docker run -p 5000:5000 elf
```

## 📚 Additional Resources

- [Project Documentation](project_documentation.md) - Comprehensive technical details
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html) - ML algorithms
- [Flask Documentation](https://flask.palletsprojects.com/) - Web framework
- [XGBoost Guide](https://xgboost.readthedocs.io/) - Gradient boosting
- [Pandas Documentation](https://pandas.pydata.org/) - Data manipulation

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**: Provide clear description and testing evidence

### Code Standards
- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation accordingly

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 📧 Contact & Support

**Author**: [Your Name]  
**Email**: uzzal.220605@s.pust.ac.bd  
**Institution**: Pabna University of Science and Technology  
**GitHub**: [@yourusername](https://github.com/yourusername)

**Questions or Feedback?**
- 📝 Open a GitHub Issue
- 💬 Start a Discussion
- 📮 Send an email

---

## 🎓 Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{electric_load_2024,
  title={Electric Load Forecasting System: Advanced ML Solutions for Energy Demand Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/electric-load-forecasting}
}
```

---

<div align="center">

**Made with ❤️ for the energy analytics community**

[⬆ Back to top](#-electric-load-forecasting-system)

</div> 