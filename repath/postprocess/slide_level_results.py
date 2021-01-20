from abc import ABC, abstractmethod

from repath.postprocess.instance_segmentors import ConnectedComponents, DBScan
from repath.postprocess.results import SlidePatchSetResults


class SlideClassifier(ABC):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    @abstractmethod
    def calc_features(self, result) -> pd.DataFrame:
        pass


class SlideClassifierWang(SlideClassifier):
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

    def calculate_slide_features(self, result: SlidePatchSetResults, outcols: List, posname: str = 'tumor'):
        print(f'calculating features for {result_post.slide_path.stem}')

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
            outvals = self.get_global_features(heatmap, labelled_image, tissue_area)
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
                outvals = self.get_region_features(reg)
            else:
                outvals = [0] * 10
            loc_list[0, (rg * 10):((rg + 1) * 10)] = outvals

        # combine global features and lesion features into one array
        features_list = np.hstack((glob_list, loc_list))

        # convert to dataframe with column names
        features_df = pd.DataFrame(features_list, columns=out_cols[:-1])
        features_df['filename'] = result.slide_path.stem

        return features_df

    def calc_features(self, results: SlidesIndexResults) -> pd.DataFrame:
        
        # create column names
        out_cols = ["area_ratio_5", "prob_score_5", "area_ratio_6", "prob_score_6", "area_ratio_7", "prob_score_7",
                    "area_ratio_8", "prob_score_8", "area_ratio_9", "prob_score_9",
                    "area_1", "eccentricity_1", "extent_1", "bbox_area_1", "major_axis_1", "max_intensity_1",
                    "mean_intensity_1", "min_intensity_1", "aspect_ratio_1", "solidity_1",
                    "area_2", "eccentricity_2", "extent_2", "bbox_area_2", "major_axis_2", "max_intensity_2",
                    "mean_intensity_2", "min_intensity_2", "aspect_ratio_2", "solidity_2", "filename"]
        features_all_slides = []

        for result in results:
            features_out = self.calculate_slide_features(result, outcols)
            features_all_slides.append(features_out)

        features_all_slides = pd.DataFrame(features_all_slides, columns=outcols)

        # write out lesions
        self.output_dir.mkdir(parents=True, exist_ok=True)
        features_all_slides.to_csv(self.output_dir / 'features.csv', index=False)



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
