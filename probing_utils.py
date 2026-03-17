import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_linear_svm(hidden_states_benign, hidden_states_harmful, c_value=0.0002, random_state=42):
    """
    hidden_states_benign: torch.Tensor (N, D)
    hidden_states_harmful: torch.Tensor (N, D)
    """

    # 1. 转 numpy
    benign = hidden_states_benign.detach().cpu().numpy()
    harmful = hidden_states_harmful.detach().cpu().numpy()

    # 2. 构造数据
    X = np.concatenate([benign, harmful], axis=0)  # (200, 4096)
    y = np.concatenate([
        np.ones(len(benign)),
        np.zeros(len(harmful))
    ])  # benign=1, harmful=0

    # 3. 划分 train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=random_state,
        stratify=y   # 保证类别平衡
    )

    # 4. 训练 Linear SVM
    clf = LinearSVC(
        C=c_value,
        max_iter=10000,
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # 5. 测试 accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 6. 取 linear weight
    # clf.coef_ shape: (1, D)
    weight_np = clf.coef_.flatten()  # (D,)
    # bias = clf.intercept_[0]

    # ---- 转 torch，用 float32 做归一化 ----
    weight = torch.from_numpy(weight_np).to(torch.float32)
    # bias = torch.tensor(bias_np, dtype=torch.float32)

    norm = weight.norm()  # 这里一般不会是 0
    weight_unit_f32 = weight / (norm + 1e-12)
    # bias_unit_f32 = bias / (norm + 1e-12)
    weight_unit_f16 = weight_unit_f32.to(torch.float16)

    return {
        "probing_direction": weight_unit_f16,
        "accuracy": acc
    }