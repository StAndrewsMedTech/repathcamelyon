from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import pandas as pd
from repath.data.annotations import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.openslide import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve


class Camelyon16(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)

    def load_annotations(self, file: Path) -> AnnotationSet:
        group_labels = {"Tumor": "tumor", "_0": "tumor", "_1": 'tumor', "_2": 'normal'}
        annotations = load_annotations(file, group_labels) if file else []
        labels_order = ["background", "tumor", "normal"]
        return AnnotationSet(annotations, self.labels, labels_order, "normal")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "normal": 1, "tumor": 2}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"normal": 0, "tumor": 1}


def training():
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    tumor_slide_dir = root / "tumor"
    normal_slide_dir = root / "normal"

    # all paths are relative to the dataset 'root'
    annotation_paths = sorted([p.relative_to(root) for p in annotations_dir.glob("*.xml")])
    tumor_slide_paths = sorted([p.relative_to(root) for p in tumor_slide_dir.glob("*.tif")])
    normal_slide_paths = sorted([p.relative_to(root) for p in normal_slide_dir.glob("*.tif")])

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] = annotation_paths + ["" for _ in range(len(normal_slide_paths))]
    df["label"] = ['tumor'] * len(tumor_slide_paths) + ['normal'] * len(normal_slide_paths)
    df["tags"] = ""

    return Camelyon16(root, df)


def training_small():
    # set up the paths to the slides and annotations
    cam16 = training()
    df = cam16.paths.sample(12, random_state=777)

    return Camelyon16(project_root() / cam16.root, df)



def testing():
    # TODO: Add this
    pass


def calc_slide_metrics(slide_results, ci=True, nreps=1000):
    # Accuracy - number of matching labels / total number of slides

    slide_accuracy = np.sum(slide_results.true_label == slide_results.predictions) / slide_results.shape[0]
    # ROC curve for camelyon 16

    conf_mat16 = conf_mat_raw(slide_results.true_label.to_numpy(),
                                  slide_results.predictions.to_numpy(),
                                  labels=["normal", "tumor"])
    conf_mat16 = conf_mat16.ravel().tolist()

    slide_results_out = [slide_accuracy] + conf_mat16
    slide_results_out = np.reshape(slide_results_out, (1, 5))

    if ci:
        slide_accuracy1000 = []
        auc_cam1000 = []
        conf_mat1000 = np.empty((nreps, 4))

        for rep in range(nreps):
            sample_slide_results = slide_results.sample(frac=1.0, replace=True)
            slide_accuracy = np.sum(sample_slide_results.true_label == sample_slide_results.predictions) / \
                             sample_slide_results.shape[0]
            slide_accuracy1000.append(slide_accuracy)
            conf_mat16 = conf_mat_raw(sample_slide_results.true_label.to_numpy(),
                                          sample_slide_results.predictions.to_numpy(), labels=["normal", "tumor"])
            conf_mat16 = conf_mat16.ravel().tolist()
            conf_mat1000[rep, :] = conf_mat16

        accuracyCI = np.reshape(np.quantile(slide_accuracy1000, [0.025, 0.975]), (2, 1))
        confmatCI = np.quantile(conf_mat1000, [0.025, 0.975], axis=0)

        slide_results_out = np.vstack((slide_results_out, np.hstack((accuracyCI, confmatCI))))

    return slide_results_out


def calc_slide_curve(slide_results, results_dir, prefix, ci=True, nreps=1000):

    # ROC curve for camelyon 16
    tumor_probs = [float(prob) for prob in slide_results.tumor.tolist()]

    fpr, tpr, roc_thresholds = roc_curve(slide_results.true_label.tolist(),
                                         tumor_probs,
                                         pos_label='tumor')
    auc_cam16 = auc(fpr, tpr)
    auc_out = np.reshape(auc_cam16, (1,1))

    if ci:
        auc_cam1000 = []
        fpr_levels = np.linspace(0, 1, 101)
        fpr_levels = fpr_levels[1:]
        tpr1000 = np.empty((nreps, 100))

        for rep in range(nreps):
            sample_slide_results = slide_results.sample(frac=1.0, replace=True)
            tumor_probs = [float(prob) for prob in sample_slide_results.tumor.tolist()]
            fpr, tpr, roc_thresholds = roc_curve(sample_slide_results.true_label.tolist(), tumor_probs,
                                                 pos_label='tumor')
            auc_cam16 = auc(fpr, tpr)
            auc_cam1000.append(auc_cam16)
            tpr_lev = np.interp(fpr_levels, fpr, tpr)
            tpr1000[rep, :] = tpr_lev

        aucCI = np.reshape(np.quantile(auc_cam1000, [0.025, 0.975]), (2, 1))
        tprCI = np.quantile(tpr1000, [0.025, 0.975], axis=0)

        auc_out = np.vstack((auc_out, aucCI))

        # write out the curve information
        # fpr, tpr, roc_thresholds - slide level roc curve for c16
        slide_curve = pd.DataFrame(np.hstack((tprCI.T, np.reshape(fpr_levels, (100, 1)))),
                                   columns=['tpr_lower', 'tpr_upper', 'fpr'])
        slide_curve.to_csv(results_dir / (prefix + '_slide_curve_CI.csv'), index=False)
        slide_curve_plt = plotROCCI(fpr, tpr, fpr_levels, tprCI, auc_cam16, aucCI.tolist(),
                                    "Receiver Operating Characteristic Curve for Slide Classification",
                                    "False Positive Rate", "True Positive Rate")
        slide_curve_plt.savefig(results_dir / (prefix + "_slide_curve_CI.png"))

    slide_curve = pd.DataFrame(list(zip(fpr, tpr, roc_thresholds)),
                               columns=['fpr', 'tpr', 'roc_thresholds'])
    slide_curve.to_csv(results_dir / (prefix + '_slide_curve.csv'), index=False)
    slide_curve_plt = plotROC(fpr, tpr, auc_cam16,
                              "Receiver Operating Characteristic Curve for Slide Classification",
                              "False Positive Rate", "True Positive Rate")
    slide_curve_plt.savefig(results_dir / (prefix + "_slide_curve.png"))

    return auc_out
