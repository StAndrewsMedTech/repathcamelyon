import numpy as np
import torch

from repath.preprocess.patching import SlidePatchSet




def evaluate_loop_dp(model, device, loader, num_classes):

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    num_samples = len(loader) * loader.batch_size

    prob_out = np.zeros((num_samples, num_classes))
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data, target = batch
            data = data.to(device)
            output = model(data)
            sm = torch.nn.Softmax(1)
            output_sm = sm(output)
            pred_prob = output_sm.cpu().numpy()  # rows: batch_size, cols: num_classes

            start = idx * loader.batch_size
            end = start + pred_prob.shape[0]
            prob_out[start:end, :] = pred_prob

            if idx % 100 == 0:
                print('Batch {} of {}'.format(idx, len(loader)))

    return prob_out



def inference_on_slide(slide_dataset: SlidePatchSet, model: torch.nn.Module, num_classes: int,
                       batch_size: int, num_workers: int, ntransforms: int = 1) -> np.array:

    """ runs inference for every patch on a slide using data parallel

    Outputs probabilities for each class

    Args:
        slide_dataset: A SlideDataset object containing all non background patches for the slide
        model: a patch classifier model
        num_classes: the number of output classes predicted by the model
        batch_size: the batch size for inference
        num_workers: the num_workers for inference
        ntransforms: the number of predictions per patch. Each patch can be predicted multiple times eg rotations
            or flips, the mean across thes transforms is found for each patch

    Returns:
        An ndarray the same length as the slide dataset with a column for each class containing a float that
        represents the probability of the patch being that class.
        
    """

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader = torch.utils.data.DataLoader(slide_dataset, shuffle=False,
                                              batch_size=batch_size,  num_workers=num_workers)

    probabilities = evaluate_loop_dp(model, device, test_loader, num_classes)

    npreds = int(len(slide_dataset) * ntransforms)

    probabilities = probabilities[0:npreds, :]

    if ntransforms > 1:
        prob_rows = probabilities.shape[0]
        prob_rows = int(prob_rows / ntransforms)
        probabilities_reshape = np.empty((prob_rows, num_classes))
        for cl in num_classes:
            class_probs = probabilities[:, cl]
            class_probs = np.reshape(class_probs, (ntransforms, prob_rows)).T
            class_probs = np.mean(class_probs, axis=1)
            probabilities_reshape[:, cl] = class_probs
        probabilities = probabilities_reshape

    return probabilities




### Below is initial pseudo code on ddp needs developing to speed up inference
#def process_predict(rank, loader_for_subset, model, batch_size, num_classes):
#    model = model.to_gpu(rank)
#    output = Tensor.empty(len(loader_for_subset) * batch_size, num_classes))
#    for batch_idx, batch in enumerate(loader_for_subset):
#        batch = batch.to_gpu(rank)
#        y = model(batch)
#        y = nn.softmax(y)
#        output[batch_idx * batch_size, :] = y
#    return output
#
#
#torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')#
#
#
#def distibuted_predict(model, patchset):
#    dataset = SlideDataset(patchset)
#    sampler = SequentialSampler(dataset)
