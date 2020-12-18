import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Tuple

from repath.preprocess.patching import SlidesIndex


def split_camelyon16(index: SlidesIndex, train_percent: float, seed:int = 5678) -> Tuple[SlidesIndex, SlidesIndex]:
    print("Splitting Camelyon 16")

    # get summaries of total patch classes for each slide
    summaries = index.summary()

    # count numbers of each slide label
    slide_label_dict = index.dataset.slide_labels
    slide_numbers_total = {}
    slide_numbers_train = {}
    slide_numbers_valid = {}
    for key in slide_label_dict.keys():
        n_slides = np.sum(summaries.slide_label == key)
        slide_numbers_total[key] = n_slides
        slide_numbers_train[key] = int(n_slides * train_percent)
        slide_numbers_valid[key] = n_slides - int(n_slides * train_percent)

    #print("slide_numbers_train", slide_numbers_train)
    #print("slide_numbers_valid", slide_numbers_valid)

    # divide each class of slide according to train percent
    slide_numbers = np.array(list(range(len(index.dataset))))
    np.random.seed(seed)
    normal_slides = slide_numbers[index.dataset.paths.label == 'normal']
    #print("normal_slides", normal_slides)
    normal_slides_train = np.random.choice(normal_slides, slide_numbers_train['normal'], replace=False)
    #print("normal_slides_train", normal_slides_train)
    tumor_slides = slide_numbers[index.dataset.paths.label == 'tumor']
    #print("tumor_slides", tumor_slides)
    tumor_slides_train = np.random.choice(tumor_slides, slide_numbers_train['tumor'], replace=False)
    #print("tumor_slides_train", tumor_slides_train)
    train_slide_numbers = np.hstack((normal_slides_train, tumor_slides_train))
    #print("train_slide_numbers", train_slide_numbers)
    valid_slide_numbers = [item for item in slide_numbers if item not in train_slide_numbers]
    #print("valid_slide_numbers", valid_slide_numbers)
    train_index = [index[idx] for idx in train_slide_numbers]
    valid_index = [index[idx] for idx in valid_slide_numbers]

    # sum total number of tumor patches in each set
    train_summaries = summaries.iloc[train_slide_numbers, :]
    valid_summaries = summaries.iloc[valid_slide_numbers, :]
    total_tumor_patches = np.sum(summaries.tumor)
    train_tumor_patches = np.sum(train_summaries.tumor)
    valid_tumor_patches = np.sum(valid_summaries.tumor)

    # some sort of check and adjustment if massively different to train_percent
    # training tumor patches should be train percent or close to it
    # close is defined as plus or minus 5% of train percent
    # so if train percent is 70, 10% of 70 is 7 so close is 70 plus or minus 3.5
    train_percent_5 = train_percent * 0.05
    train_low = (train_percent - train_percent_5) * total_tumor_patches
    train_high = (train_percent + train_percent_5) * total_tumor_patches
    if not (train_low <= train_tumor_patches <= train_high):
        print("Warning: splitting on slides has not got a balanced split of patches ")
    if train_tumor_patches == 0:
        train_slide_numbers = train_slide_numbers + [tumor_slides[0]]
        valid_slide_numbers = slide_numbers[~train_slide_numbers]
        train_index = index[train_slide_numbers]
        valid_index = index[valid_slide_numbers]

    return SlidesIndex(index.dataset, train_index), SlidesIndex(index.dataset, valid_index)


def split_camelyon17(index: SlidesIndex, train_percent: float, seed:int = 5678) -> Tuple[SlidesIndex, SlidesIndex]:

    # get summaries of total patch classes for each slide
    summaries = index.summary()

    # count numbers of each slide label
    slide_label_dict = index.dataset.slide_labels
    slide_numbers_total = {}
    slide_numbers_train = {}
    slide_numbers_valid = {}
    for key in slide_label_dict.keys():
        n_slides = np.sum(summaries.slide_label == key)
        slide_numbers_total[key] = n_slides
        slide_numbers_train[key] = int(n_slides * train_percent)
        slide_numbers_valid[key] = n_slides - int(n_slides * train_percent)

    # find annotated slides
    annotated_slides = []
    unannotated_slides = []
    annotated_patients = []
    all_patients = []
    annotated_or_not = []
    for idx, patchset in enumerate(index):
        ps_tags = patchset.tags.split(',')
        ps_tags = [tg.strip() for tg in ps_tags]
        if 'annotated' in ps_tags:
            annotated_slides.append(idx)
            # get patients in annotated slides
            annotated_patients.append(ps_tags[0])
            annotated_or_not.append(True)
        else:
            unannotated_slides.append(idx)
            annotated_or_not.append(False)
        all_patients.append(ps_tags[0])

    # split patients into train and valid
    unique_annotated_patients = np.unique(annotated_patients).tolist()
    unique_all_patients = np.unique(all_patients).tolist()
    unique_unannotated_patients = unique_all_patients[~unique_annotated_patients]

    n_annotated_patients = len(unique_annotated_patients)
    n_train_annotated_patients = int(train_percent * n_annotated_patients)
    n_all_patients = len(unique_all_patients)
    n_train_all_patients = int(train_percent * n_all_patients)
    n_train_unannotated = n_train_all_patients - n_train_annotated_patients

    annotated_patients_train = np.random.choice(unique_annotated_patients, n_train_annotated_patients, replace=False)
    unannotated_patients_train = np.random.choice(unique_unannotated_patients, n_train_unannotated, replace=False)
    train_patients = annotated_patients_train.tolist() + unannotated_patients_train.tolist()
    valid_patients = unique_all_patients[~train_patients]
    train_patients_mask = [True if pt in train_patients else False for pt in all_patients]
    valid_patients_mask = [not pt for pt in train_patients_mask]

    # split slides into train and valid
    train_index = index[valid_patients_mask]
    valid_index = index[valid_patients_mask]

    # check split of patch classes
    train_summaries = summaries[train_patients_mask]
    valid_summaries = summaries[valid_patients_mask]
    total_tumor_patches = np.sum(summaries.tumor)
    train_tumor_patches = np.sum(train_summaries.tumor)
    valid_tumor_patches = np.sum(valid_summaries.tumor)

    # some sort of check and adjustment if massively different to train_percent
    # training tumor patches should be train percent or close to it
    # close is defined as plus or minus 5% of train percent
    # so if train percent is 70, 10% of 70 is 7 so close is 70 plus or minus 3.5
    train_percent_5 = train_percent * 0.05
    train_low = (train_percent - train_percent_5) * total_tumor_patches
    train_high = (train_percent + train_percent_5) * total_tumor_patches
    if not (train_low <= train_tumor_patches <= train_high):
        print("Warning: splitting on patients has not got a balanced split of patches")

    # check split of slide classes
    slide_class_split = pd.DataFrame(index=['train', 'valid', 'total'], columns=['negative', 'itc', 'micro', 'macro'])
    for col in slide_class_split.columns:
        slide_class_split.iloc['train', col] = np.sum(train_summaries.slide_label == col)
        slide_class_split.iloc['valid', col] = np.sum(valid_summaries.slide_label == col)
        slide_class_split.iloc['total', col] = np.sum(summaries.slide_label == col)
    if (np.sum(np.less(slide_class_split.tonumpy(), 1)) > 0):
        print("Warning: splitting on patients has some slide classes in train or valid as zero")

    return train_index, valid_index


def select_annotated(index: SlidesIndex) -> SlidesIndex:
    annotated = [ps for ps in index if 'annotated' in ps.tags]
    return SlidesIndex(index.dataset, annotated)