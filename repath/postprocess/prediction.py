


from repath.postprocess.slide_dataset import SlideDataset


def process_predict(rank, loader_for_subset, model, batch_size, num_classes):
    model = model.to_gpu(rank)
    output = Tensor.empty(len(loader_for_subset) * batch_size, num_classes))
    for batch_idx, batch in enumerate(loader_for_subset):
        batch = batch.to_gpu(rank)
        y = model(batch)
        y = nn.softmax(y)
        output[batch_idx * batch_size, :] = y
    return output


torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')


def distibuted_predict(model, patchset):
    dataset = SlideDataset(patchset)
    sampler = SequentialSampler(dataset)
    