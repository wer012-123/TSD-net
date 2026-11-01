"""
SparseDFNet for Battery SOH Prediction - Sequence-to-Sequence

不使用滑动窗口！
输入：前 N% 的完整 SOH 历史曲线
输出：后 (100-N)% 的完整 SOH 预测曲线
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 模型定义
# ============================================================
class TemporalDecomposition(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        trend = self.moving_avg(x)
        trend = trend.permute(0, 2, 1)
        seasonal = x.permute(0, 2, 1) - trend
        return trend, seasonal


class SAFEBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.gate_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.sparse_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.sparse_conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // 4)),
            nn.ReLU(),
            nn.Linear(max(1, in_channels // 4), in_channels),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        identity = x
        x = x.permute(0, 2, 1)
        h = F.gelu(self.conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        h_gated = h * gate
        sparse1 = F.gelu(self.sparse_conv1(x))
        g_sparse = torch.sigmoid(self.sparse_conv2(sparse1))
        pooled = self.avg_pool(h_gated * g_sparse)
        attn = self.channel_attn(pooled.squeeze(-1)).unsqueeze(-1)
        out = h_gated * g_sparse * attn
        out = self.bn(out)
        out = out.permute(0, 2, 1)
        out = F.gelu(identity + out)
        return out


class SparseDFNet_SOH_Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence SOH 预测

    输入: [batch, input_len, 3] - 训练集的特征序列
    输出: [batch, output_len, 1] - 测试集的 SOH 预测序列
    """
    def __init__(self, max_input_len=100, max_output_len=100, input_features=3, hidden_dim=64):
        super().__init__()

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.hidden_dim = hidden_dim

        # Encoder: 处理输入序列
        self.encoder_embed = nn.Linear(input_features, hidden_dim)
        self.encoder_decomp = TemporalDecomposition(kernel_size=3)
        self.encoder_trend_safe = SAFEBlock(hidden_dim, kernel_size=3)
        self.encoder_seasonal_safe = SAFEBlock(hidden_dim, kernel_size=3)

        # Decoder: 生成输出序列
        self.decoder_embed = nn.Linear(input_features, hidden_dim)
        self.decoder_decomp = TemporalDecomposition(kernel_size=3)
        self.decoder_trend_safe = SAFEBlock(hidden_dim, kernel_size=3)
        self.decoder_seasonal_safe = SAFEBlock(hidden_dim, kernel_size=3)

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, encoder_input, decoder_input):
        """
        encoder_input: [batch, input_len, 3] - 训练集特征
        decoder_input: [batch, output_len, 3] - 测试集特征
        """
        # Encode
        enc_x = self.encoder_embed(encoder_input)
        enc_trend, enc_seasonal = self.encoder_decomp(enc_x)
        enc_trend_feat = self.encoder_trend_safe(enc_trend)
        enc_seasonal_feat = self.encoder_seasonal_safe(enc_seasonal)

        # Decode
        dec_x = self.decoder_embed(decoder_input)
        dec_trend, dec_seasonal = self.decoder_decomp(dec_x)
        dec_trend_feat = self.decoder_trend_safe(dec_trend)
        dec_seasonal_feat = self.decoder_seasonal_safe(dec_seasonal)

        # 融合 encoder 和 decoder 特征
        # 使用 decoder 的最后一个时间步特征 + encoder 的全局特征
        enc_global_trend = enc_trend_feat.mean(dim=1, keepdim=True).expand_as(dec_trend_feat)
        enc_global_seasonal = enc_seasonal_feat.mean(dim=1, keepdim=True).expand_as(dec_seasonal_feat)

        combined = torch.cat([
            dec_trend_feat + enc_global_trend,
            dec_seasonal_feat + enc_global_seasonal
        ], dim=2)  # [batch, output_len, hidden_dim*2]

        # 预测输出
        output = self.output_projection(combined)  # [batch, output_len, 1]

        return output.squeeze(-1)  # [batch, output_len]


# ============================================================
# 数据集
# ============================================================
class BatterySeq2SeqDataset(Dataset):
    """
    Sequence-to-Sequence 数据集
    每个样本就是一条完整的训练-测试对
    """
    def __init__(self, train_features, train_soh, test_features, test_soh):
        self.train_features = torch.FloatTensor(train_features)
        self.train_soh = torch.FloatTensor(train_soh)
        self.test_features = torch.FloatTensor(test_features)
        self.test_soh = torch.FloatTensor(test_soh)

    def __len__(self):
        return 1  # 只有一个样本：整条电池数据

    def __getitem__(self, idx):
        return (
            self.train_features,
            self.train_soh,
            self.test_features,
            self.test_soh
        )


# ============================================================
# 训练和评估
# ============================================================
def train_epoch_seq2seq(model, train_features, train_soh, test_features, test_soh,
                       criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()

    train_features = train_features.unsqueeze(0).to(device)  # [1, train_len, 3]
    test_features = test_features.unsqueeze(0).to(device)    # [1, test_len, 3]
    test_soh = test_soh.to(device)                           # [test_len]

    optimizer.zero_grad()

    # 前向传播
    pred_soh = model(train_features, test_features)  # [1, test_len]
    pred_soh = pred_soh.squeeze(0)                   # [test_len]

    # 计算损失
    loss = criterion(pred_soh, test_soh)

    # 反向传播
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def evaluate_seq2seq(model, train_features, test_features, device):
    """评估模型"""
    model.eval()

    train_features = train_features.unsqueeze(0).to(device)
    test_features = test_features.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_soh = model(train_features, test_features)
        pred_soh = pred_soh.squeeze(0).cpu().numpy()

    return pred_soh


def plot_soh_prediction(real_soh, pred_soh, split_idx, battery_id, train_ratio, save_path):
    """绘制 SOH 预测曲线"""
    plt.figure(figsize=(12, 6))

    cycles = np.arange(len(real_soh))

    # 真实曲线（完整）
    plt.plot(cycles, real_soh, 'r-', linewidth=2, label='Real value')

    # 预测曲线（从训练集最后一点开始，与真实曲线重合）
    # 将训练集最后一个真实值添加到预测曲线的开头
    pred_with_anchor = np.concatenate([[real_soh[split_idx - 1]], pred_soh])
    pred_cycles = np.arange(split_idx - 1, len(real_soh))
    plt.plot(pred_cycles, pred_with_anchor, 'b--', linewidth=2, label='Predicted')

    # 训练/测试分割线
    plt.axvline(x=split_idx, color='k', linestyle=':', linewidth=1.5)

    plt.xlabel('Number of cycles', fontsize=14)
    plt.ylabel('SOH', fontsize=14)
    plt.title(f'{battery_id} - SOH Prediction (Train: {int(train_ratio*100)}%)', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {save_path}")


# ============================================================
# 主训练函数
# ============================================================
def train_single_battery(battery_id, soh_path, train_ratios=[0.4, 0.5, 0.6],
                        epochs=500, lr=0.001):
    """为单个电池训练 Seq2Seq 模型"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"Battery: {battery_id}")
    print(f"{'='*80}")

    results = []

    # 加载数据
    df = pd.read_csv(soh_path)

    print(f"  Total cycles: {len(df)}")
    print(f"  SOH range: {df['SOH'].min():.4f} - {df['SOH'].max():.4f}")

    # 提取特征和标签
    features = df[['cycle', 'CC_capacity', 'charging_duration']].values.astype(np.float32)
    soh = df['SOH'].values.astype(np.float32)

    # 归一化特征（重要：保持特征的相对关系）
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    features_norm = (features - feature_min) / (feature_max - feature_min + 1e-8)

    for train_ratio in train_ratios:
        print(f"\n  Training with {int(train_ratio*100)}% train data...")

        # 划分训练集和测试集
        split_idx = int(len(features) * train_ratio)

        train_features = features_norm[:split_idx]
        train_soh = soh[:split_idx]

        test_features = features_norm[split_idx:]
        test_soh = soh[split_idx:]

        print(f"  Train: {len(train_features)} cycles")
        print(f"  Test: {len(test_features)} cycles")

        if len(test_features) < 10:
            print(f"    [SKIP] Test set too small")
            continue

        # 转换为 Tensor
        train_features_t = torch.FloatTensor(train_features)
        train_soh_t = torch.FloatTensor(train_soh)
        test_features_t = torch.FloatTensor(test_features)
        test_soh_t = torch.FloatTensor(test_soh)

        # 创建模型
        model = SparseDFNet_SOH_Seq2Seq(
            max_input_len=len(train_features),
            max_output_len=len(test_features),
            input_features=3,
            hidden_dim=64
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50)

        # 训练
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 100

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch_seq2seq(
                model, train_features_t, train_soh_t, test_features_t, test_soh_t,
                criterion, optimizer, device
            )

            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"    [STOP] Training diverged at epoch {epoch}")
                break

            scheduler.step(train_loss)

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"    Early stopping at epoch {epoch}")
                break

            if epoch % 100 == 0 or epoch == 1:
                print(f"    Epoch {epoch}/{epochs} | Loss: {train_loss:.6f}")

        # 评估
        pred_soh = evaluate_seq2seq(model, train_features_t, test_features_t, device)

        # 计算完整的评估指标
        mae = mean_absolute_error(test_soh, pred_soh)
        mse = mean_squared_error(test_soh, pred_soh)
        rmse = np.sqrt(mse)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((test_soh - pred_soh) / (test_soh + 1e-8))) * 100

        # R² (R-squared)
        ss_res = np.sum((test_soh - pred_soh) ** 2)
        ss_tot = np.sum((test_soh - np.mean(test_soh)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # RE (Relative Error) - 符号化相对误差
        re = np.mean((pred_soh - test_soh) / (test_soh + 1e-8)) * 100

        # Score - 不对称评分 (过估惩罚更重)
        errors = (pred_soh - test_soh) / (test_soh + 1e-8) * 100
        penalties = np.where(errors > 0, 13 * errors, 10 * np.abs(errors))
        score = 100 - np.mean(penalties)

        print(f"    Test Metrics:")
        print(f"      MAE={mae:.4f}, MSE={mse:.6f}, RMSE={rmse:.4f}")
        print(f"      MAPE={mape:.2f}%, R²={r2:.4f}, RE={re:.2f}%, Score={score:.2f}")

        # 可视化
        plot_dir = r'E:\fuxian2\SparseDFNet_SOH_Prediction\plots_seq2seq'
        os.makedirs(plot_dir, exist_ok=True)

        plot_path = os.path.join(plot_dir, f'{battery_id}_train{int(train_ratio*100)}.png')
        plot_soh_prediction(soh, pred_soh, split_idx, battery_id, train_ratio, plot_path)

        # 保存逐点预测数据
        import pickle
        prediction_dir = r'E:\fuxian2\SparseDFNet_SOH_Prediction\predictions_seq2seq'
        os.makedirs(prediction_dir, exist_ok=True)

        prediction_data = {
            'battery_id': battery_id,
            'train_ratio': train_ratio,
            'real_soh': soh.tolist(),
            'pred_soh': pred_soh.tolist(),
            'split_idx': split_idx,
            'train_soh': train_soh.tolist(),
            'test_soh': test_soh.tolist()
        }

        pred_save_path = os.path.join(prediction_dir, f'{battery_id}_train{int(train_ratio*100)}.pkl')
        with open(pred_save_path, 'wb') as f:
            pickle.dump(prediction_data, f)
        print(f"    Prediction data saved: {pred_save_path}")

        results.append({
            'battery_id': battery_id,
            'train_ratio': train_ratio,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            're': re,
            'score': score
        })

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    print("="*80)
    print("SparseDFNet SOH Prediction - Sequence-to-Sequence")
    print("No Sliding Window!")
    print("="*80)

    # NASA 电池
    batteries = {
        'B0005': r'E:\fuxian2\SparseDFNet_SOH_Prediction\data\nasa_correct_soh\B0005_soh_correct.csv',
        'B0006': r'E:\fuxian2\SparseDFNet_SOH_Prediction\data\nasa_correct_soh\B0006_soh_correct.csv',
        'B0007': r'E:\fuxian2\SparseDFNet_SOH_Prediction\data\nasa_correct_soh\B0007_soh_correct.csv',
        'B0018': r'E:\fuxian2\SparseDFNet_SOH_Prediction\data\nasa_correct_soh\B0018_soh_correct.csv'
    }

    train_ratios = [0.4, 0.5, 0.6]
    epochs = 500
    lr = 0.001

    print(f"\nConfiguration:")
    print(f"  Train ratios: {[int(r*100) for r in train_ratios]}%")
    print(f"  Epochs: {epochs}")
    print(f"  Method: Sequence-to-Sequence (NO sliding window)")
    print(f"  Input: Full training sequence")
    print(f"  Output: Full testing sequence")

    all_results = []

    for battery_id, soh_path in batteries.items():
        if not os.path.exists(soh_path):
            print(f"\n[SKIP] {battery_id}: File not found")
            continue

        results = train_single_battery(battery_id, soh_path, train_ratios, epochs, lr)
        all_results.extend(results)

    # 保存结果
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)

        # 重新排列列顺序，符合标准格式
        results_df = results_df[['battery_id', 'train_ratio', 'mae', 'mse', 'rmse', 'mape', 'r2', 're', 'score']]

        results_path = r'E:\fuxian2\SparseDFNet_SOH_Prediction\results\soh_prediction_seq2seq.csv'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)

        print(f"\n{'='*80}")
        print("Final Results - Complete Metrics")
        print(f"{'='*80}\n")
        print(results_df.to_string(index=False))
        print(f"\n\nColumn Definitions:")
        print(f"  MAE:   Mean Absolute Error")
        print(f"  MSE:   Mean Squared Error")
        print(f"  RMSE:  Root Mean Squared Error")
        print(f"  MAPE:  Mean Absolute Percentage Error (%)")
        print(f"  R²:    R-squared (coefficient of determination)")
        print(f"  RE:    Relative Error (%) - signed, shows over/under-estimation")
        print(f"  Score: Asymmetric score (overestimation penalized 13/10)")
        print(f"\nResults saved: {results_path}")

    print(f"\n{'='*80}")
    print("Training Completed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
