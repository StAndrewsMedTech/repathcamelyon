from abc import ABC, abstractmethod
from joblib import dump, load
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import label, regionprops
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from xgboost import XGBClassifier

from repath.postprocess.instance_segmentors import ConnectedComponents, DBScan
from repath.postprocess.results import SlidesIndexResults, SlidePatchSetResults
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, fpr_tpr_curve, save_conf_mat_plot, save_conf_mat_plot_ci, binary_curves


class SlideClassifier(ABC):
    def __init__(self, slide_labels: Dict) -> None:
        self.slide_labels = slide_labels

    @abstractmethod
    def calculate_slide_features(self, result) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict_slide_level(self) -> None:
        pass

    def calc_features(self, results: SlidesIndexResults, output_dir: Path) -> pd.DataFrame:
        
        features_all_slides = []

        print(results[0])

        for result in results:
            features_out = self.calculate_slide_features(result)
            features_all_slides.append(features_out)

        outcols = features_all_slides[0].columns.values

        features_all_slides = pd.concat(features_all_slides)

        # write out lesions
        output_dir.mkdir(parents=True, exist_ok=True)
        features_all_slides.to_csv(output_dir / 'features.csv', index=False)
  

    def calc_slide_metrics_binary(self, title, output_dir, ci=True, nreps=1000, posname='tumor'):
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_results = pd.read_csv(output_dir / 'slide_results.csv')

        # Accuracy - number of matching labels / total number of slides
        slide_accuracy = np.sum(slide_results.true_label == slide_results.predictions) / slide_results.shape[0]
        print(f'accuracy: {slide_accuracy}')

        # confusion matrix
        conf_mat = conf_mat_raw(slide_results.true_label.to_numpy(),
                                    slide_results.predictions.to_numpy(),
                                    labels=self.slide_labels.keys())
        conf_mat = conf_mat.ravel().tolist()

        # ROC curve for 
        pos_probs = [float(prob) for prob in slide_results[posname].tolist()]
        precision, tpr, fpr, roc_thresholds = binary_curves(slide_results.true_label.to_numpy(),
                                                            np.array(pos_probs), pos_label=posname)
        auc_out = auc(tpr, fpr)

        # write out precision recall curve - without CI csv and png
        slide_curve = pd.DataFrame(list(zip(fpr, tpr, roc_thresholds)), 
                                   columns=['fpr', 'tpr', 'thresholds'])
        slide_curve.to_csv(output_dir / 'slide_pr_curve.csv', index=False)
        title_pr = "Slide Classification Precision-Recall Curve for \n" + title
        pr_curve_plt = plotROC(tpr, fpr, auc_out, title_pr, "True Positive Rate", "False Positive Rate")
        pr_curve_plt.savefig(output_dir / 'slide_pr_curve.png')

        # combine single figure results into dataframe
        slide_metrics_out = [slide_accuracy] + conf_mat + [auc_out]
        slide_metrics_out = np.reshape(slide_metrics_out, (1, 6))
        slide_metrics_out = pd.DataFrame(slide_metrics_out, columns=['accuray', 'tn', 'fp', 'fn', 'tp', 'auc'])
        slide_metrics_out.index = ['results']

        # create confidence matrix plot and write out
        title_cm = "Slide Classification Confusion Matrix for \n" + title
        save_conf_mat_plot(slide_metrics_out[['tn', 'fp', 'fn', 'tp']], self.slide_labels.keys(), title_cm, output_dir)

        if ci:
            slide_accuracy1000 = np.empty((nreps, 1))
            conf_mat1000 = np.empty((nreps, 4))
            auc_out1000 = np.empty((nreps, 1))
            nrecall_levs = 101
            tpr_levels = np.linspace(0.0, 1.0, nrecall_levs)
            fpr1000 = np.empty((nreps, nrecall_levs))
            for rep in range(nreps):
                sample_slide_results = slide_results.sample(frac=1.0, replace=True)
                slide_accuracy = np.sum(sample_slide_results.true_label == sample_slide_results.predictions) / \
                                sample_slide_results.shape[0]
                slide_accuracy1000[rep, 0] = slide_accuracy
                conf_mat = conf_mat_raw(sample_slide_results.true_label.to_numpy(),
                                            sample_slide_results.predictions.to_numpy(), labels=self.slide_labels.keys())
                conf_mat = conf_mat.ravel().tolist()
                conf_mat1000[rep, :] = conf_mat

                #pos_probs = [float(prob) for prob in sample_slide_results[posname].tolist()]
                fpr_lev = fpr_tpr_curve(sample_slide_results.true_label.to_numpy(), sample_slide_results[posname].to_numpy(),
                                                        pos_label=posname, recall_levels=tpr_levels)
                auc_samp = auc(tpr_levels[1:], fpr_lev[1:])
                auc_out1000[rep, 0] = auc_samp
                #tpr_lev = np.interp(fpr_levels, fpr, tpr)
                fpr1000[rep, :] = fpr_lev
            
            # combine single figure metrics to dataframe
            samples_df = pd.DataFrame(np.hstack((slide_accuracy1000, conf_mat1000, auc_out1000)), 
                                      columns=['accuray', 'tn', 'fp', 'fn', 'tp', 'auc'])
            samples_df.index = ['sample_' + str(x) for x in range(nreps)]
            slide_metrics_ci = samples_df.quantile([0.025, 0.975])
            slide_metrics_ci.index = ['ci_lower_bound', 'ci_upper_bound']
            slide_metrics_out = pd.concat((slide_metrics_out, slide_metrics_ci, samples_df), axis=0)

            # create confidence matrix plot with confidence interval and write out
            save_conf_mat_plot_ci(slide_metrics_out[['tn', 'fp', 'fn', 'tp']], self.slide_labels.keys(), title_cm, output_dir)

            # write out the curve information
            fprCI = np.quantile(fpr1000, [0.025, 0.975], axis=0)
            # fpr, tpr, roc_thresholds - slide level roc curve for c16
            slide_curve = pd.DataFrame(np.hstack((fprCI.T, np.reshape(tpr_levels, (nrecall_levs, 1)))),
                                       columns=['fpr_lower', 'fpr_upper', 'tpr'])
            slide_curve.to_csv(output_dir / 'slide_pr_curve_ci.csv', index=False)
            slide_curve_plt = plotROCCI(tpr, fpr, tpr_levels, fprCI, auc_out, slide_metrics_ci.auc.tolist(),
                                        title_pr, "True Positive Rate", "False Positive Rate")
            slide_curve_plt.savefig(output_dir / "slide_pr_curve_ci.png")

        slide_metrics_out.to_csv(output_dir / 'slide_metrics.csv')

    def calc_slide_metrics_multi(self, title, output_dir, ci=True, nreps=1000):
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_results = pd.read_csv(output_dir / 'slide_results.csv')

        # Accuracy - number of matching labels / total number of slides
        slide_accuracy = np.sum(slide_results.true_label == slide_results.predictions) / slide_results.shape[0]
        print(f'accuracy: {slide_accuracy}')
        
        # confusion matrix for multi class
        conf_mat = conf_mat_raw(slide_results.true_label.to_numpy(),
                                    slide_results.predictions.to_numpy(),
                                    labels=self.slide_labels.keys())
        conf_mat = conf_mat.ravel().tolist()
        pred_tiled_labels = list(self.slide_labels.keys()) * len(self.slide_labels.keys())
        true_tiled_labels = [item for item in self.slide_labels.keys() for i in range(len(self.slide_labels.keys()))]
        confmat_labels = [f'true_{vals[0]}_pred_{vals[1]}' for vals in list(zip(true_tiled_labels, pred_tiled_labels))]
        column_labels = ['accuracy'] + confmat_labels

        output_list = [slide_accuracy] + conf_mat
        output_arr = np.reshape(np.array(output_list), (1, len(output_list)))
        slide_metrics_out = pd.DataFrame(output_arr)
        slide_metrics_out.columns = column_labels
        slide_metrics_out.index = ['results']

        # create confidence matrix plot and write out
        title_cm = "Slide Classification Confusion Matrix for \n" + title
        save_conf_mat_plot(slide_metrics_out.iloc[:, 1:], self.slide_labels.keys(), title_cm, output_dir)

        if ci:
            slide_accuracy1000 = np.empty((nreps, 1))
            conf_mat1000 = np.empty((nreps, len(self.slide_labels.keys())**2))
            for rep in range(nreps):
                sample_slide_results = slide_results.sample(frac=1.0, replace=True)
                slide_accuracy = np.sum(sample_slide_results.true_label == sample_slide_results.predictions) / \
                                sample_slide_results.shape[0]
                slide_accuracy1000[rep, 0] = slide_accuracy
                conf_mat = conf_mat_raw(sample_slide_results.true_label.to_numpy(),
                                            sample_slide_results.predictions.to_numpy(), labels=self.slide_labels.keys())
                conf_mat = conf_mat.ravel().tolist()
                conf_mat1000[rep, :] = conf_mat

            # combine single figure metrics to dataframe
            samples_df = pd.DataFrame(np.hstack((slide_accuracy1000, conf_mat1000)), 
                                      columns=column_labels)
            samples_df.index = ['sample_' + str(x) for x in range(nreps)]
            slide_metrics_ci = samples_df.quantile([0.025, 0.975])
            slide_metrics_ci.index = ['ci_lower_bound', 'ci_upper_bound']
            slide_metrics_out = pd.concat((slide_metrics_out, slide_metrics_ci, samples_df), axis=0)

            # create confidence matrix plot with confidence interval and write out
            save_conf_mat_plot_ci(slide_metrics_out.iloc[:, 1:], self.slide_labels.keys(), title_cm, output_dir)

        slide_metrics_out.to_csv(output_dir / 'slide_metrics.csv')

    def calc_slide_metrics(self, title, output_dir, ci=True, nreps=1000, posname='tumor'):
        if len(self.slide_labels) == 2:
            self.calc_slide_metrics_binary(title, output_dir, ci=True, nreps=1000, posname='tumor')
        else:
            self.calc_slide_metrics_multi(title, output_dir, ci=True, nreps=1000)


class SlideClassifierWang(SlideClassifier):
    def calculate_slide_features(self, result: SlidePatchSetResults, posname: str = 'tumor') -> pd.DataFrame:
        def get_global_features(img_grey: np.array, labelled_image: np.ndarray, tissue_area: int) -> Tuple[float, float]:
            """ Create features based on whole slide properties of a given trheshold

            Args:
                img_grey: a greyscale heatmap where each pixel represents a patch. pixel values from 0, 255
                    with 255 representing a probability of one
                thresh: the threshold probability to use to bbinarize the image
                tiss_area: the area of the image that is tissue in pixels

            Returns:
                A tuple containing:
                    area_ratio - the ratio of number of pixels over the given probability to the tissue area
                    prob_area - the sum of probability of all the pixels over the threshold divided by the tissue area
            """

            # measure connected components
            reg_props_t = regionprops(labelled_image)
            # get area for each region
            img_areas = [reg.area for reg in reg_props_t]
            # get total area of tumor regions
            metastatic_area = np.sum(img_areas)
            # get area ratio
            area_ratio = metastatic_area / tissue_area

            # get list of regions
            labels_t = np.unique(labelled_image)
            # create empty list of same size
            lab_list_t = np.zeros((len(labels_t), 1))
            # for each region
            for lab in range(1, len(labels_t)):
                # get a mask of just that region
                mask = labelled_image == lab
                # sum the probability over the region in the mask
                tot_prob = np.sum(np.divide(img_grey[mask], 255))
                # add to empty list
                lab_list_t[lab, 0] = tot_prob
            # sum over whole list
            tot_prob_t = np.sum(lab_list_t)
            # diveide by tissue area
            prob_area = tot_prob_t / tissue_area

            return area_ratio, prob_area

        def get_region_features(reg) -> list:
            """ Get list of properties of a ragion

            Args:
                reg: a region from regionprops function

            Returns:
                A list of 11 region properties

            """
            # get area of region
            reg_area = reg.area
            # eccentricity - for an ellipse with same second moments as region
            # divide distance between focal points by length of major axis
            reg_eccent = reg.eccentricity
            # extent ratio of pixels in region to pixels in bounding box
            reg_extent = reg.extent
            # area of bounding box of region
            reg_bbox_area = reg.bbox_area
            # major axis length of ellipse with same second moment of area
            reg_maj_ax_len = reg.major_axis_length
            # highest probabaility in the region
            reg_max_int = reg.max_intensity
            # mean probability voer he region
            reg_mean_int = reg.mean_intensity
            # lowest probability in the region
            reg_min_int = reg.min_intensity
            # Rrtio of pixels in the region to pixels of the convex hull image.
            reg_solid = reg.solidity
            # cacluate aspect ration of region bounding box
            reg_bbox = reg.bbox
            reg_aratio = (reg_bbox[2] - reg_bbox[0]) / (reg_bbox[3] - reg_bbox[1])

            output_list = [reg_area, reg_eccent, reg_extent, reg_bbox_area, reg_maj_ax_len, reg_max_int,
                        reg_mean_int, reg_min_int, reg_aratio, reg_solid]
            return output_list

    
        print(f'calculating features for {result.slide_path.stem}')

        heatmap = result.to_heatmap(posname)
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # area of tissue is the number of rows in results dataframe
        tissue_area = result.patches_df.shape[0]

        # set thresholds for global features
        threshz = [0.5, 0.6, 0.7, 0.8, 0.9]

        # create storage for global features as an 1xn array. There are two features for each threshold.
        glob_list = np.zeros((1, 2 * len(threshz)))
        # for each threshold calculate two features and store in array
        for idx, th in enumerate(threshz):
            segmentor = ConnectedComponents(th)
            labelled_image = segmentor.segment(heatmap)
            outvals = get_global_features(heatmap, labelled_image, tissue_area)
            glob_list[0, (idx * 2)] = outvals[0]
            glob_list[0, (idx * 2 + 1)] = outvals[1]

        # get two largest areas at 0.5 thresh
        segmentor = ConnectedComponents(0.5)
        labelled_image = segmentor.segment(heatmap)

        # measure connected components
        reg_props_5 = regionprops(labelled_image, intensity_image=heatmap)

        # get area for each region
        img_areas_5 = [reg.area for reg in reg_props_5]

        # get labels for each region
        img_label_5 = [reg.label for reg in reg_props_5]

        # sort in descending order
        toplabels = [x for _, x in sorted(zip(img_areas_5, img_label_5), reverse=True)][0:2]

        # create empty 1x20 array to store ten feature values each for top 2 lesions
        loc_list = np.zeros((1, 20))

        # per lesion add to store - labels start from 1 need to subtract 1 for zero indexing
        for rg in range(2):
            if len(img_areas_5) > rg:
                reg = reg_props_5[toplabels[rg] - 1]
                outvals = get_region_features(reg)
            else:
                outvals = [0] * 10
            loc_list[0, (rg * 10):((rg + 1) * 10)] = outvals

        # combine global features and lesion features into one array
        features_list = np.hstack((glob_list, loc_list))

        # create column names
        out_cols = ["area_ratio_5", "prob_score_5", "area_ratio_6", "prob_score_6", "area_ratio_7", "prob_score_7",
                    "area_ratio_8", "prob_score_8", "area_ratio_9", "prob_score_9",
                    "area_1", "eccentricity_1", "extent_1", "bbox_area_1", "major_axis_1", "max_intensity_1",
                    "mean_intensity_1", "min_intensity_1", "aspect_ratio_1", "solidity_1",
                    "area_2", "eccentricity_2", "extent_2", "bbox_area_2", "major_axis_2", "max_intensity_2",
                    "mean_intensity_2", "min_intensity_2", "aspect_ratio_2", "solidity_2"]

        # convert to dataframe with column names
        features_df = pd.DataFrame(features_list, columns=out_cols)
        features_df['slidename'] = result.slide_path.stem
        features_df['slide_label'] = result.label.lower()
        features_df['tags'] = result.tags

        return features_df

    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']

        # fit or load (NB not all experiments will fit a sepearate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ['true_label']
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)


class SlideClassifierLee(SlideClassifier):

    def calculate_slide_features(self, result: SlidePatchSetResults, posname: str = 'tumor') -> pd.DataFrame:
        def get_region_features(reg) -> list:
            """ Get list of properties of a ragion

            Args:
                reg: a region from regionprops function

            Returns:
                A list of 8 region properties

            """
            # get area of region
            reg_area = reg.area
            # major_axis_length of a regoin
            reg_major_axis = reg.major_axis_length
            # minor_axis_length of a region
            reg_minor_axis = reg.minor_axis_length
            # density of a region
            reg_density = 1 / reg_area
            # mean, max , min  probability of a region
            reg_mean_intensity = reg.mean_intensity
            reg_max_intensity = reg.max_intensity
            reg_min_intensity = reg.min_intensity

            output_list = [reg_area, reg_major_axis, reg_minor_axis, reg_density, reg_mean_intensity, reg_max_intensity,
                        reg_min_intensity]

            return output_list

        print(f'calculating features for {result.slide_path.stem}')

        heatmap = result.to_heatmap(posname)
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # get two largest areas at 0.5 thresh
        segmentor = DBScan(0.58, eps=3, min_samples=20)
        labelled_image = segmentor.segment(heatmap)
        labelled_image = np.array(labelled_image, dtype='int')

        # measure connected components
        reg_props = regionprops(labelled_image, intensity_image=heatmap)

        # get area for each region
        img_areas = [reg.area for reg in reg_props]

        # get labels for each region
        img_label = [reg.label for reg in reg_props]

        # sort in descending order
        toplabels = [x for _, x in sorted(zip(img_areas, img_label), reverse=True)][0:3]

        # create empty 1x8 array to store 7 feature values each for top 3 lesions
        feature_list = np.zeros((1, 21))

        # labels in image are nto zero indexed reg props are so need to adjust for non zero indexing
        for rg in range(3):
            if len(img_areas) > rg:
                toplab = toplabels[rg]
                topindex = img_label.index(toplab)
                reg = reg_props[topindex]
                outvals = get_region_features(reg)
            else:
                outvals = [0] * 7
            feature_list[0, (rg * 7):((rg + 1) * 7)] = outvals

        out_cols = ["major_axis_1", "minor_axis_1", "area_1", "density_1", "mean_probability_1", "max_probability_1",
                    "min_probability_1", "major_axis_2", "minor_axis_2", "area_2", "density_2", "mean_probability_2", 
                    "max_probability_2", "min_probability_2", "major_axis_3", "minor_axis_3", "area_3", "density_3", 
                    "mean_probability_3", "max_probability_3", "min_probability_3"]
        # convert to dataframe with column names
        features_df = pd.DataFrame(feature_list, columns=out_cols)
        features_df['slidename'] = result.slide_path.stem
        features_df['slide_label'] = result.label
        features_df['tags'] = result.tags

        return features_df


    def predict_slide_level(self, features_dir: Path, classifier_dir: Path, retrain=False):
        classifier_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        labels = [lab.lower() for lab in labels]
        names = features['slidename']
        tags = features['tags']

        # fit or load (NB not all experiments will fit a separate model so fitting bundled in with predict)
        if Path(classifier_dir, 'slide_model.joblib').is_file() and not retrain:
            slide_model = load(classifier_dir / 'slide_model.joblib')
        else:
            slide_model = XGBClassifier()
            slide_model.fit(just_features, labels)
            dump(slide_model, classifier_dir / 'slide_model.joblib')

        just_features = just_features.astype(np.float)
        predictions = slide_model.predict(just_features)
        # Probabilities for each class
        probabilities = slide_model.predict_proba(just_features)
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ["true_label"]
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)


class SlideClassifierLiu(SlideClassifier):

    def calculate_slide_features(self, result: SlidePatchSetResults, posname: str = 'tumor') -> pd.DataFrame:
        print(f'calculating features for {result.slide_path.stem}')

        output_list = [np.max(result.patches_df[posname]), result.slide_path.stem, result.label, result.tags]
        output_arr = np.reshape(np.array(output_list), (1, 4))

        out_cols = ["max_probability", 'slidename', 'slide_label', 'tags']
        # convert to dataframe with column names
        feature_df = pd.DataFrame(output_arr, columns=out_cols)

        return feature_df

    def predict_slide_level(self, features_dir: Path):
        features_dir.mkdir(parents=True, exist_ok=True)
        features = pd.read_csv(features_dir / 'features.csv')

        just_features = features.drop(['slidename', 'slide_label', 'tags'], axis=1)
        labels = features['slide_label']
        names = features['slidename']
        tags = features['tags']

        just_features = just_features.astype(np.float)
        predictions = np.array(just_features.max_probability > 0.9)
        predictions = np.where(predictions, "tumor", "normal")
        # Probabilities for each class
        probabilities = np.array(just_features.max_probability)
        probabilities = probabilities.reshape((probabilities.shape[0], 1))
        probabilities = np.hstack((np.subtract(1, probabilities), probabilities))
        # combine predictions and probailities into a dataframe
        reshaped_predictions = predictions.reshape((predictions.shape[0], 1))
        preds_probs_df = pd.DataFrame(np.hstack((reshaped_predictions, probabilities)))
        print(self.slide_labels.keys())
        print(preds_probs_df)
        preds_probs_df.columns = ["predictions"] + list(self.slide_labels.keys())
        # create slide label dataframe
        slides_labels_df = pd.DataFrame(labels)
        slides_labels_df.columns = ["true_label"]
        slides_labels_df["true_label"] = [lab.lower() for lab in slides_labels_df["true_label"]]
        # create one dataframe with slide results and true labels
        slides_names_df = pd.DataFrame(names)
        slides_names_df.columns = ["slide_names"]
        tags_df = pd.DataFrame(tags)
        tags_df.columns = ["tags"]
        slide_results = pd.concat((slides_names_df, slides_labels_df, tags_df, preds_probs_df), axis=1)

        slide_results.to_csv(features_dir / 'slide_results.csv', index=False)
