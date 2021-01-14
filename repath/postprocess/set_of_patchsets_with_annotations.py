import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, confusion_matrix, cohen_kappa_score

from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve


def calc_patch_level_metrics(patches_df: pd.DataFrame, poslabel: str = 'tumor', optimal_threshold: float = 0.5) -> pd.DataFrame:
    # for poslabel column create a boolean mask of values greater than threshold (true means patch is detected as tumor)
    poslabel_mask = np.greater_equal(patches_df[poslabel], optimal_threshold)
    # get prediction for each patch which are either poslabel or other
    predictions = np.where(poslabel_true, poslabel, 'other')
    # calculate accuracy for poslabel
    patch_accuracy = np.sum(patches_df.label == patches_df.predictions) / patches_df.shape[0]
    # calculate number of true postives etc - not using scikit learn function as it is slow
    tn, fp, fn, tp = conf_mat_raw(patches_df.label.to_numpy(), patches_df.predictions.to_numpy(),
                                  labels=['other', 'tumor']).ravel()
    # calculate patch recall for poslabel
    patch_recall = tp / (tp + fn)
    # calculate patch specificity for poslabel
    patch_specificity = tn / (tn + fp)
    # calculate patch precision for poslabel
    patch_precision = tp / (tp + fp)
    # write results to list
    patch_results_out = [patch_accuracy, tn, fp, fn, tp, patch_recall, patch_specificity, patch_precision]

    return patch_results_out


def patch_level_metrics(slide_results: List[SlidesIndexResults], save_dir: Path,
                        poslabel: str = 'tumor', optimal_threshold: float = 0.5, 
                        ci: bool = False, nreps: int = 1000) -> pd.DataFrame:

    # combine into one set of patches
    all_patches = CombinedIndex.for_slide_indexes(SlidesIndexResults)
    # get dict which contains just class labels without background
    class_labels = remove_label_from_dict(all_patches.dataset.labels, 'background')  

    # get one number summaries
    patch_results_out = calc_patch_level_metrics(all_patches.patches_df, posLabel, optimal_threshold)

    # use precision recall from scikit - calculates every threshold
    patch_precisions, patch_recalls, patch_thresholds = precision_recall_curve(all_patches.label,
                                                                               all_patches.probability,
                                                                               pos_label='tumor')
    # calculate pr auc
    patch_pr_auc = auc(patch_recalls, patch_precisions)
    # add to list of results
    patch_Results_out.append(path_pr_auc)
    
    # write out precision recall curve - without CI csv and png
    patch_curve = pd.DataFrame(list(zip(patch_precisions, patch_recalls, patch_thresholds)),
                               columns=['patch_precisions', 'patch_recalls', 'patch_thresholds'])
    patch_curve.to_csv(save_dir / 'patch_pr_curve.csv', index=False)
    pr_curve_plt = plotROC(patch_recalls, patch_precisions, patch_pr_auc,
                           "Precision-Recall Curve for Patch Classification", 'Recall', 'Precision')
    pr_curve_plt.savefig(save_dir/ "patch_pr_curve.png")

    # convert list to dataframe with row name - results
    col_names = ['accuracy', 'tn', 'fp', 'fn', 'tp', 'recall', 'specificity', 'precision', 'auc']
    patch_results_out = pd.DataFrame(np.reshape(patch_results_out, (1, len(patch_results_out))), columns=col_names)
    patch_results_out.index = ['results']

    if ci:
        # create empty list to store results this will be end up as a list of list
        # more efficient to convert list of lists to pandas dataframe once than append row by row
        patch_ci = []
        # will calculate precision at a specified set of recall levels, this will be the same length for each sample
        # if used precision_recall_curve recalls and length would vary due to different numbers of thresholds
        nrecall_levs = 1001
        # create empty numpy array for storing precisions
        precisions1000 = np.empty((nreps, nrecall_levs))
        # set recall levels
        recall_levels = np.linspace(0.0, 1.0, nrecall_levs)
        for rep in range(nreps):
            # create bootstrap sample
            sample_patches = all_patches.patches_df.sample(frac=1.0, replace=True)
            # get one number summaries
            ci_results = calc_patch_level_metrics(sample_patches, col_names, posLabel, optimal_threshold)
            # get precisions and store
            pre1000 = pre_re_curve(sample_patches.label.to_numpy(), sample_patches.probability.to_numpy(), 'tumor',
                                   recall_levels)
            precisions1000[rep, :] = pre1000
            # get pr auc
            ci_pr_auc = auc(recall_levels[1:], pre1000[1:])
            # append to this set of results
            ci_results.append(ci_pr_auc)
            # append ci_results to create list of lists of ci_results
            patch_ci.append(ci_results)
 
        # convert list of lists to a dataframe
        patch_ci_df = pd.DataFrame(patch_ci, columns=col_names)

        # name to rows to give sample numbers
        patch_ci_df.index = ['sample_' + str(x) for x in range(nreps)]
        # create confidence intervals for each 
        patch_results_out_ci = patch_ci_df.quantile([0.025, 0.975])
        # rename rows of dataframe
        patch_results_out_ci.index = ['ci_lower_bound', 'ci_upper_bound']

        # concatenate results to give results, confidence interval then all samples
        patch_results_out = pd.concat((patch_results_out, patch_results_out_ci, patch_ci_df), axis=0)

        # create confidence interval for precision recall curve
        precisions_ci = np.quantile(precisions1000, [0.025, 0.975], axis=0)
        # create dataframe with precision recall curve confidence interval
        patch_curve_ci = pd.DataFrame(np.hstack((precisions_ci.T, np.reshape(recall_levels, (1001, 1)))),
                                      columns=['patch_precisions_lower', 'patch_precisions_upper', 'patch_recalls'])
        # write out precision recall curve confidence interval
        patch_curve_ci.to_csv(save_dir / 'patch_pr_curve_ci.csv', index=False)

        # create pr curve with confidence interval
        pr_curve_plt = plotROCCI(patch_recalls, patch_precisions, recall_levels, precisionsCI, patch_pr_auc,
                                 patch_pr_aucCI, "Precision-Recall Curve for Patch Classification", 'Recall',
                                 'Precision')
        pr_curve_plt.savefig(save_dir / "patch_pr_curve_ci.png")

    # write out patch summary result dataframe
    save_dir.mkdir(parents=True, exist_ok=True)
    patch_results_out.to_csv(save_dir / 'patch_results.csv')
    
 

