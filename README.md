Key Features

- **Temporal Decomposition**: Separates battery degradation into trend and seasonal components
- **Sparse Adaptive Mechanism**: Three-layer gating (Channel, Spatial, Temporal) for robust feature selection
- **Seq2Seq Architecture**: Autoregressive decoder for multi-step SOH prediction
- **Cross-Dataset Validation**: Tested on NASA and MATR battery datasets
- **Multiple Training Ratios**: Supports 40%, 50%, 60% training configurations

---

## 📊 Performance Highlights

| Metric | TSD-Net | TCN | PINN | HDLBDQ |
|--------|---------|-----|------|---------|
| **MAE** | 0.00464 | 0.00531 | 0.00714 | 0.00727 |
| **RMSE** | 0.00567 | 0.00772 | 0.01934 | 0.01241 |
| **R²** | 0.97981 | 0.92918 | 0.85079 | 0.89532 |

- **50% Win Rate** on RMSE (best robustness)
- **58.3% Win Rate** on R² (best trend fitting)
- **RMSE/MAE Ratio: 1.22** (lowest error variance)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/wer012-123/TSD-net.git
cd TSD-net
pip install -r requirements.txt
```

### Training

**NASA Dataset (B0005, B0006, B0007, B0018):**
```bash
python train_seq2seq.py
```

**MATR Dataset (b1c0-b1c7):**
```bash
python train_matr_oxford.py
```

### Evaluation

```bash
# Calculate R² scores
python calculate_r2.py

# Generate complete metrics table
python generate_complete_metrics_table.py

# Summarize all R² results
python summarize_all_r2.py
```

---

## 📁 Project Structure

```
TSD-net/
├── data/                          # Dataset preprocessing
│   ├── extract_nasa_correct_soh.py
│   ├── extract_matr_oxford_soh.py
│   └── fix_soh_data.py
├── models/                        # Model checkpoints (generated)
├── predictions_seq2seq/           # NASA predictions (PKL)
├── predictions_matr/              # MATR predictions (PKL)
├── train_seq2seq.py              # NASA training script
├── train_matr_oxford.py          # MATR training script
├── calculate_r2.py               # R² evaluation
└── README.md
```

---

## 🔬 Model Architecture

### Input Features (per cycle)
1. **Cycle Number**: Current charge-discharge cycle index
2. **CC Capacity**: Constant-current charging capacity (Ah)
3. **Charging Duration**: Time spent in charging phase (seconds)

### Output
- **SOH Scalar**: State of Health value in range [0, 1]

### Core Components
1. **Temporal Decomposition Block**: MovingAvg kernel for trend/seasonal separation
2. **ASE (Adaptive Sparse Encoder)**:
   - Channel Gating: Feature-wise importance
   - Spatial Gating: Cross-feature dependencies
   - Temporal Gating: Time-step significance
3. **Seq2Seq Decoder**: Autoregressive prediction with teacher forcing

---

## 📈 Datasets

### NASA Battery Dataset
- **Batteries**: B0005, B0006, B0007, B0018
- **Cycles**: 168 average cycles (range: 72-168)
- **EOL Criterion**: Capacity drops to 70% of rated capacity

### MATR Battery Dataset
- **Batteries**: b1c0 - b1c7 (8 cells)
- **Cycles**: 1000+ cycles (ultra-long-term)
- **EOL Criterion**: 80% capacity retention

---

## 🎯 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 500 |
| Optimizer | Adam |
| Loss Function | MSE |
| Input Window | 10 cycles |
| Prediction Horizon | 1 cycle |

---

## 📊 Visualization

Generate prediction curves with training/test split markers:

```bash
# NASA datasets with TCN comparison
python train_seq2seq_combined_plots.py

# MATR datasets
python train_matr_combined_plots.py
```

**Output**: PNG files with:
- Solid line: Ground truth SOH
- Dashed line: Predicted SOH
- Vertical line: Train/test split point

---

## 🔍 Evaluation Metrics

- **MAE** (Mean Absolute Error): Point-wise consistency
- **RMSE** (Root Mean Squared Error): Robustness to outliers
- **R²** (Coefficient of Determination): Trend fitting capability

**Win-Tie-Loss Statistics**: Pairwise comparison across 36 experimental configurations (12 batteries × 3 training ratios)

---

## 📝 Citation

If you use TSD-Net in your research, please cite:

```bibtex
@article{tsdnet2024,
  title={TSD-Net: Time-Series Decomposition Network for Battery State of Health Prediction},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [@wer012-123](https://github.com/wer012-123)

---

## 🙏 Acknowledgments

- NASA Prognostics Center of Excellence (PCoE) for battery datasets
- MATR dataset contributors
- PyTorch team for the deep learning framework

---

## 📚 Related Work

- **Autoformer**: Decomposition Transformers with Auto-Correlation
- **Informer**: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
- **TCN**: Temporal Convolutional Networks for sequence modeling

---

**⭐ If you find this project helpful, please consider giving it a star!**

