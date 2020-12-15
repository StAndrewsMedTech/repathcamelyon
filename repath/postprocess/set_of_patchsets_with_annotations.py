import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, confusion_matrix, cohen_kappa_score

from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve



def calc_patch_level_metrics(all_patches, optimal_threshold, ci=False, nreps=1000):
    # Accuracy
    all_patches['predictions'] = np.where(np.greater_equal(all_patches.probability, optimal_threshold), 'tumor',
                                          'unlabelled')
    all_patches['label'] = [patch.lower() for patch in all_patches['label']]
    patch_accuracy = np.sum(all_patches.label == all_patches.predictions) / all_patches.shape[0]
    tn, fp, fn, tp = conf_mat_raw(all_patches.label.to_numpy(), all_patches.predictions.to_numpy(),
                                      labels=['unlabelled', 'tumor']).ravel()
    # Recall
    patch_recall = tp / (tp + fn)
    # Specificity
    patch_specificity = tn / (tn + fp)
    # Precision
    patch_precision = tp / (tp + fp)

    patch_results_out = [patch_accuracy, tn, fp, fn, tp, patch_recall, patch_specificity, patch_precision]
    patch_results_out = np.reshape(patch_results_out, (1, 8))

    if ci:
        accuracy1000 = []
        tn1000 = []
        fp1000 = []
        fn1000 = []
        tp1000 = []
        recall1000 = []
        spec1000 = []
        prec1000 = []

        for rep in range(nreps):
            sample_patches = all_patches.sample(frac=1.0, replace=True)
            sample_patches['predictions'] = np.where(np.greater_equal(sample_patches.probability, optimal_threshold),
                                                     'tumor', 'unlabelled')
            sample_patches['label'] = [patch.lower() for patch in sample_patches['label']]
            patch_accuracy = np.sum(sample_patches.label == sample_patches.predictions) / sample_patches.shape[0]
            accuracy1000.append(patch_accuracy)
            tn, fp, fn, tp = conf_mat_raw(sample_patches.label.to_numpy(), sample_patches.predictions.to_numpy(),
                                          labels=['unlabelled', 'tumor']).ravel()
            tn1000.append(tn)
            fp1000.append(fp)
            fn1000.append(fn)
            tp1000.append(tp)
            # Recall
            patch_recall = tp / (tp + fn)
            recall1000.append(patch_recall)
            # Specificity
            patch_specificity = tn / (tn + fp)
            spec1000.append(patch_specificity)
            # Precision
            patch_precision = tp / (tp + fp)
            prec1000.append(patch_precision)

        patch_accuracyCI = np.quantile(accuracy1000, [0.025, 0.975])
        tnCI = np.quantile(tn1000, [0.025, 0.975])
        fpCI = np.quantile(fp1000, [0.025, 0.975])
        fnCI = np.quantile(fn1000, [0.025, 0.975])
        tpCI = np.quantile(tp1000, [0.025, 0.975])
        patch_recallCI = np.quantile(recall1000, [0.025, 0.975])
        patch_specificityCI = np.quantile(spec1000, [0.025, 0.975])
        patch_precisionCI = np.quantile(prec1000, [0.025, 0.975])

        patch_results_out_ci = np.hstack((np.reshape(patch_accuracyCI, (2, 1)), np.reshape(tnCI, (2, 1)),
            np.reshape(fpCI, (2, 1)), np.reshape(fnCI, (2, 1)), np.reshape(tpCI, (2, 1)),
            np.reshape(patch_recallCI, (2, 1)), np.reshape(patch_specificityCI, (2, 1)),
            np.reshape(patch_precisionCI, (2, 1)))
        patch_results_out = np.vstack((patch_results_out, patch_results_out_ci))

    return patch_results_out


def calc_patch_level_curve(all_patches, results_dir, suffix, ci=False, nreps=1000):

    patch_precisions, patch_recalls, patch_thresholds = precision_recall_curve(all_patches.label,
                                                                               all_patches.probability,
                                                                               pos_label='tumor')
    patch_pr_auc = auc(patch_recalls, patch_precisions)

    patch_curve = pd.DataFrame(list(zip(patch_precisions, patch_recalls, patch_thresholds)),
                               columns=['patch_precisions', 'patch_recalls', 'patch_thresholds'])
    patch_curve.to_csv(results_dir / (suffix + 'patch_curve.csv'), index=False)
    pr_curve_plt = plotROC(patch_recalls, patch_precisions, patch_pr_auc,
                           "Precision-Recall Curve for Patch Classification", 'Recall', 'Precision')
    pr_curve_plt.savefig(results_dir / (suffix + "patch_curve.png"))

    patch_results_out = [patch_pr_auc]
    patch_results_out = np.reshape(patch_results_out, (1,1))

    if ci:
        auc1000 = []
        precisions1000 = np.empty((nreps, 1001))
        recall_levels = np.linspace(0.0, 1.0, 1001)
        for rep in range(nreps):
            sample_patches = all_patches.sample(frac=1.0, replace=True)
            # PR AUC
            pre1000 = pre_re_curve(sample_patches.label.to_numpy(), sample_patches.probability.to_numpy(), 'tumor',
                                   recall_levels)
            patch_pr_auc = auc(recall_levels[1:], pre1000[1:])
            auc1000.append(patch_pr_auc)
            precisions1000[rep, :] = pre1000

        patch_pr_aucCI = np.quantile(auc1000, [0.025, 0.975])
        precisionsCI = np.quantile(precisions1000, [0.025, 0.975], axis=0)

        patch_curve = pd.DataFrame(np.hstack((precisionsCI.T, np.reshape(recall_levels, (1001, 1)))),
                                   columns=['patch_precisions_lower', 'patch_precisions_upper', 'patch_recalls'])
        patch_curve.to_csv(results_dir / (suffix + 'patch_curve_CI.csv'), index=False)

        pr_curve_plt = plotROCCI(patch_recalls, patch_precisions, recall_levels, precisionsCI, patch_pr_auc,
                                 patch_pr_aucCI, "Precision-Recall Curve for Patch Classification", 'Recall',
                                 'Precision')
        pr_curve_plt.savefig(results_dir / (suffix + "patch_curve_withCI.png"))

        patch_results_out = np.vstack((patch_results_out, np.reshape(patch_pr_aucCI, (2, 1))))

    return patch_results_out

