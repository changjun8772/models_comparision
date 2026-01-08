import pickle
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


class _ProbabilityCalibrator:

    def __init__(self, method: str = 'platt'):
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")

        y_true = np.array(y_true).ravel()
        y_prob = np.array(y_prob).ravel()

        if self.method == 'platt':
            # Platt Scaling
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)

        elif self.method == 'isotonic':
            # Isotonic Regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)

        else:
            raise ValueError("method must be 'platt' or 'isotonic'")

        self.is_fitted = True
        print(f"Fit completed, using {self.method}")

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("please use fit()")
        y_prob = np.array(y_prob).ravel()
        if self.method == 'platt':
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.transform(y_prob)

    def evaluate_calibration(self, y_true: np.ndarray, raw_probs: np.ndarray,
                             calibrated_probs: np.ndarray = None,
                             plot: bool = True) -> dict:
        y_true = np.array(y_true).ravel()
        raw_probs = np.array(raw_probs).ravel()

        if calibrated_probs is None:
            if not self.is_fitted:
                raise ValueError("校准器尚未拟合")
            calibrated_probs = self.calibrate(raw_probs)
        else:
            calibrated_probs = np.array(calibrated_probs).ravel()

        # calculating Brier
        raw_brier = brier_score_loss(y_true, raw_probs)
        cal_brier = brier_score_loss(y_true, calibrated_probs)

        # calculating ECE (Expected Calibration Error)
        raw_ece = self._calculate_ece(y_true, raw_probs)
        cal_ece = self._calculate_ece(y_true, calibrated_probs)

        # other metrics
        raw_predictions = (raw_probs > 0.5).astype(int)
        cal_predictions = (calibrated_probs > 0.5).astype(int)

        raw_accuracy = accuracy_score(y_true, raw_predictions)
        cal_accuracy = accuracy_score(y_true, cal_predictions)

        metrics = {
            'raw_brier': raw_brier,
            'calibrated_brier': cal_brier,
            'brier_improvement': (raw_brier - cal_brier) / raw_brier * 100,
            'raw_ece': raw_ece,
            'calibrated_ece': cal_ece,
            'ece_improvement': (raw_ece - cal_ece) / raw_ece * 100,
            'raw_accuracy': raw_accuracy,
            'calibrated_accuracy': cal_accuracy,
            'accuracy_change': cal_accuracy - raw_accuracy
        }

        print(f"Brier - before: {raw_brier:.4f}, after: {cal_brier:.4f}")
        print(f"Brier improved: {metrics['brier_improvement']:.1f}%")
        print(f"ECE - before: {raw_ece:.4f}, after: {cal_ece:.4f}")
        print(f"ECE improved: {metrics['ece_improvement']:.1f}%")
        print(f"accuracy - before: {raw_accuracy:.4f}, after: {cal_accuracy:.4f}")
        print(f"change in accuracy: {metrics['accuracy_change']:.4f}")

        if plot:
            self._plot_calibration_comparison(y_true, raw_probs, calibrated_probs)

        return metrics

    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        calculate Expected Calibration Error (ECE)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:
                bin_prob = np.mean(y_prob[mask])
                bin_acc = np.mean(y_true[mask])
                bin_weight = np.sum(mask) / len(y_true)
                ece += bin_weight * np.abs(bin_acc - bin_prob)

        return ece

    def _plot_calibration_comparison(self, y_true: np.ndarray, raw_probs: np.ndarray,
                                     calibrated_probs: np.ndarray) -> None:
        raw_prob_true, raw_prob_pred = calibration_curve(y_true, raw_probs, n_bins=10)
        cal_prob_true, cal_prob_pred = calibration_curve(y_true, calibrated_probs, n_bins=10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(raw_prob_pred, raw_prob_true, 's-', label='Original probability', color='red', markersize=6)
        ax1.plot(cal_prob_pred, cal_prob_true, 'o-', label='Calibrated probability', color='blue', markersize=4)
        ax1.plot([0, 1], [0, 1], 'k--', label='real calibration', alpha=0.5)
        ax1.set_xlabel('Predicted probability')
        ax1.set_ylabel('Ture probability')
        ax1.set_title('Curve of probability calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(raw_probs, bins=20, alpha=0.7, label='Original probability', color='red', density=True)
        ax2.hist(calibrated_probs, bins=20, alpha=0.7, label='Calibrated probability', color='blue', density=True)
        ax2.set_xlabel('Predicted probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("校准器尚未拟合")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrator': self.calibrator,
                'is_fitted': self.is_fitted
            }, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.method = data['method']
        self.calibrator = data['calibrator']
        self.is_fitted = data['is_fitted']


class PostTrainingCalibrator:
    """
    训练后校准器 - 专门为已训练模型设计
    """

    def __init__(self, trained_model):
        self.trained_model = trained_model
        self.calibrator = None

    def calibrate_using_test_split(self, x_test: np.ndarray, y_test: np.ndarray,
                                   test_size: float = 0.3, method: str = 'platt',
                                   random_state: int = 42) -> Tuple[_ProbabilityCalibrator, dict, tuple]:
        x_calib, x_eval, y_calib, y_eval = train_test_split(
            x_test, y_test,
            test_size=test_size,
            random_state=random_state,
            stratify=y_test
        )

        raw_probs_calib = self._get_prediction_probs(x_calib)
        self.calibrator = _ProbabilityCalibrator(method=method)
        self.calibrator.fit(y_calib, raw_probs_calib)

        raw_probs_eval = self._get_prediction_probs(x_eval)
        calibrated_probs_eval = self.calibrator.calibrate(raw_probs_eval)
        metrics = self.calibrator.evaluate_calibration(
            y_eval, raw_probs_eval, calibrated_probs_eval
        )
        return self.calibrator, metrics, (x_eval, y_eval)

    def predict_proba(self, X: np.ndarray, use_calibration: bool = True) -> np.ndarray:
        raw_probs = self._get_prediction_probs(X)
        if use_calibration and self.calibrator is not None and self.calibrator.is_fitted:
            return self.calibrator.calibrate(raw_probs)
        else:
            return raw_probs

    def predict(self, X: np.ndarray, threshold: float = 0.5,
                use_calibration: bool = True) -> np.ndarray:
        probabilities = self.predict_proba(X, use_calibration)
        return (probabilities > threshold).astype(int)

    def detailed_evaluation(self, X: np.ndarray, y_true: np.ndarray,
                            threshold: float = 0.5) -> dict:
        raw_probs = self._get_prediction_probs(X)
        calibrated_probs = self.predict_proba(X, use_calibration=True)

        raw_predictions = (raw_probs > threshold).astype(int)
        calibrated_predictions = (calibrated_probs > threshold).astype(int)
        results = {
            'raw': {
                'probabilities': raw_probs,
                'predictions': raw_predictions,
                'accuracy': accuracy_score(y_true, raw_predictions),
                'confusion_matrix': confusion_matrix(y_true, raw_predictions),
                'classification_report': classification_report(y_true, raw_predictions, output_dict=True)
            },
            'calibrated': {
                'probabilities': calibrated_probs,
                'predictions': calibrated_predictions,
                'accuracy': accuracy_score(y_true, calibrated_predictions),
                'confusion_matrix': confusion_matrix(y_true, calibrated_predictions),
                'classification_report': classification_report(y_true, calibrated_predictions, output_dict=True)
            }
        }
        # 打印对比结果
        print("=== 详细评估结果 ===")
        print(f"准确率 - 原始: {results['raw']['accuracy']:.4f}, 校准后: {results['calibrated']['accuracy']:.4f}")
        print(f"准确率变化: {results['calibrated']['accuracy'] - results['raw']['accuracy']:.4f}")

        print("\n原始预测混淆矩阵:")
        print(results['raw']['confusion_matrix'])

        print("\n校准后预测混淆矩阵:")
        print(results['calibrated']['confusion_matrix'])

        return results

    def _get_prediction_probs(self, X: np.ndarray) -> np.ndarray:
        """
        获取原始预测概率
        """

        raw_predictions = self.trained_model.predict(X)

        # 处理不同的预测输出格式
        if isinstance(raw_predictions, (list, np.ndarray)):
            if len(raw_predictions) > 0 and isinstance(raw_predictions[0], (list, np.ndarray)):
                return np.array([p[0] for p in raw_predictions])
            else:
                return np.array(raw_predictions)
        else:
            raise ValueError("不支持的预测输出格式")

    def save_calibrator(self, filepath: str) -> None:
        """
        保存校准器
        """
        if self.calibrator is None:
            raise ValueError("没有可保存的校准器")
        self.calibrator.save(filepath)

    def load_calibrator(self, filepath: str) -> None:
        """
        加载校准器
        """
        self.calibrator = _ProbabilityCalibrator()
        self.calibrator.load(filepath)
