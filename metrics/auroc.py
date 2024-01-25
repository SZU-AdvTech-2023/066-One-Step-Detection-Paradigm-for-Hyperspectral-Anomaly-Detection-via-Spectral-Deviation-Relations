from gc import get_threshold
import logging
import matplotlib.pyplot as plt
from numpy import ndarray as NDArray
from sklearn.metrics import roc_auc_score, roc_curve
import os
import random
import numpy as np
from scipy import integrate
from tqdm import tqdm
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import sys

sys.path.append("D:/PapersCode/TDD-master-zt/RSAD/metrics/")


def compute_auroc(epoch: int, ep_reconst, ep_gt, working_dir: str, image_level=False, save_image=True, ) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    Args:
        epoch (int): Current epoch
        ep_reconst (NDArray): Reconstructed images in a current epoch  当前轮次的重建图像
        ep_gt (NDArray): Ground truth masks in a current epoch  当前轮次的真实标签（ground truth）
    Returns:
        float: AUROC score
    """
    save_dir = os.path.join(working_dir, "epochs-" + str(epoch))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    y_score, y_true = [], []  # 分别用于存储模型的预测得分和真实标签
    for i, (amap, gt) in enumerate(tqdm(zip(ep_reconst, ep_gt))):  # amap 是重建图像，gt 是对应的真实标签
        anomaly_scores = amap[np.where(gt == 0)]  # 提取异常样本（真实标签为 0）的模型预测得分
        normal_scores = amap[np.where(gt == 1)]  # 提取正常样本（真实标签为 1）的模型预测得分
        y_score += anomaly_scores.tolist()  # 将异常样本的模型预测得分添加到 y_score 列表中。这是为了在后续评估中使用模型对异常样本的预测得分。
        y_true += np.zeros(len(anomaly_scores)).tolist()  # 将与异常样本对应的真实标签（0，表示异常）添加到 y_true 列表中。这是为了与异常样本的预测得分一一对应。
        y_score += normal_scores.tolist()  # 将正常样本的模型预测得分添加到 y_score 列表中。这是为了在后续评估中使用模型对正常样本的预测得分。
        y_true += np.ones(len(normal_scores)).tolist()  # 将与正常样本对应的真实标签（1，表示正常）添加到 y_true 列表中。这是为了与正常样本的预测得分一一对应。

    # 计算AUC的值，AUC是ROC曲线下的面积（Area Under the ROC Curve）
    scoreDF = roc_auc_score(y_true, y_score)
    # 计算 ROC（Receiver Operating Characteristic）曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)  # 得到fpr, tpr, thresholds（阈值）
    """
    True Positive Rate ( TPR )  = TP / [ TP + FN] ，TPR代表能将正例分对的概率,召回率，召回率越高越好
    False Positive Rate( FPR ) = FP / [ FP + TN] ，FPR代表将负例错分为正例的概率
    """

    """
    scoreDt，scoreDF，scoreTD，scoreODP 越大，代表检测性能越好
    scoreFt 越小 或者 scoreBS 越大，代表背景抑制性能越好
    """
    scoreDt = abs(integrate.trapz(tpr, thresholds))  # 表示在 ROC 曲线上TPR与阈值之间的面积。
    scoreFt = abs(integrate.trapz(fpr, thresholds))  # 表示在 ROC 曲线上FPR 与阈值之间的面积。
    scoreTD = scoreDF + scoreDt  # 探测器的目标可探测性
    scoreBS = scoreDF - scoreFt  # 探测器的背景抑制性
    scoreODP = scoreDF + scoreDt - scoreFt  # 总体探测概率
    scoreTDBS = scoreDt - scoreFt
    scoreSNPR = scoreDt / scoreFt   # 信噪比
    logging.info("scoreDF: " + str(scoreDF))
    logging.info("scoreDt: " + str(scoreDt))
    logging.info("scoreFt: " + str(scoreFt))
    logging.info("scoreTD: " + str(scoreTD))
    logging.info("scoreBS: " + str(scoreBS))
    logging.info("scoreODP: " + str(scoreODP))
    logging.info("scoreTDBS: " + str(scoreTDBS))
    logging.info("scoreSNPR: " + str(scoreSNPR))

    if save_image:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {round(scoreDF, 3)}")
        plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
        plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()

    return scoreDF
