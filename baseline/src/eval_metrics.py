from sklearn.metrics import roc_curve
import numpy as np


# Lowest distance between false positive and false negative
def eer_metric(scores,labels):
    f_pos_r, t_pos_r, thresholds = roc_curve(labels, scores)
    f_neg_r = 1 - t_pos_r

    distance = np.absolute(f_neg_r - f_pos_r)
    index = np.argmin(distance)
    eer_threshold = thresholds[index]
    eer = f_pos_r[index]

    return eer, eer_threshold


# Lowest actual mistake
# DCF = C_false_negative * P_false_negative * P_target + C_false_positive * P_false_positive * (1 - P_target)
def minDCF_metric(scores,labels):
    p_target = 0.01  #Probability of the same speaker
    c_fn = 1 #Cost of false_neg
    c_fp = 1 #Cost of false_pos
    f_pos_r, t_pos_r, thresholds = roc_curve(labels, scores)
    f_neg_r = 1 - t_pos_r

    min_dcf = float("inf")
    threshold = None

    for i in range(len(thresholds)):
        dcf = c_fn * f_neg_r[i] * p_target + c_fp * f_pos_r[i] * (1 - p_target)

        if dcf < min_dcf:
            min_dcf = dcf
            threshold = thresholds[i]

    return min_dcf, threshold
