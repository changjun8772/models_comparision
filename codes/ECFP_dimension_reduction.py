import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr


def clean_ecfp(X, var_th=1e-5, corr_th=0.95):
    """
    X: np.ndarray (n_samples, 2048)  二进制/整数型 ECFP4
    return
        X_clean: 清洗后的 ECFP
        mask:    bool 数组，True 表示保留的列，可持久化供新样本使用
    """
    # 1. 方差过滤
    var_sel = VarianceThreshold(threshold=var_th)
    X_var = var_sel.fit_transform(X)
    mask_var = var_sel.get_support()  # 中间掩码

    # 2. Pearson 冗余过滤
    cols_to_keep = np.arange(X_var.shape[1])
    drop_set = set()
    for i in range(X_var.shape[1]):
        if i in drop_set:
            continue
        for j in range(i + 1, X_var.shape[1]):
            if j in drop_set:
                continue
            if abs(pearsonr(X_var[:, i], X_var[:, j])[0]) > corr_th:
                drop_set.add(j)  # 只删 j，保留 i
    keep = np.array([c for c in cols_to_keep if c not in drop_set])
    X_clean = X_var[:, keep]

    # 3. 合并两次掩码 → 原始 2048 维度的最终掩码
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[mask_var] = True
    final_mask = np.zeros(X.shape[1], dtype=bool)
    final_mask[np.where(mask_var)[0][keep]] = True
    return X_clean, final_mask


# ========== 使用示例 ==========
# X_train 形状 (n_samples, 2048)，未经降维的原始的ECFP (2, 2048)
X_ecfp_clean, ecfp_mask = clean_ecfp(fps_train)

print(f"原始维度: {fps_train.shape[1]} → 清洗后: {X_ecfp_clean.shape[1]}")

# 保存掩码
#pd.DataFrame({'mask':boruta2.support_}).to_csv("data/graphy_mask.csv", index=False)