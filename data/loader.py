""" Object detection loader/collate

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.utils.data
from effdet.anchors import AnchorLabeler
from timm.data.distributed_sampler import OrderedDistributedSampler
import os

from .transforms import *
from .random_erasing import RandomErasing

MAX_NUM_INSTANCES = 100


class DetectionFastCollate:
    """ A detection specific, optimized collate function w/ a bit of state.

    Optionally performs anchor labelling. Doing this here offloads some work from the
    GPU and the main training process thread and increases the load on the dataloader
    threads.

    """
    def __init__(
            self,
            instance_keys=None,
            instance_shapes=None,
            instance_fill=-1,
            max_instances=MAX_NUM_INSTANCES,
            anchor_labeler=None,
    ):
        instance_keys = instance_keys or {'bbox', 'bbox_ignore', 'cls', 'difficult', 'truncated', 'occluded'}
        instance_shapes = instance_shapes or dict(
            bbox=(max_instances, 4), bbox_ignore=(max_instances, 4), cls=(max_instances,), difficult=(max_instances,), truncated=(max_instances,), occluded=(max_instances,))
        self.instance_info = {k: dict(fill=instance_fill, shape=instance_shapes[k]) for k in instance_keys}
        self.max_instances = max_instances
        self.anchor_labeler = anchor_labeler

    def __call__(self, batch):
        batch_size = len(batch)
        target = dict()
        labeler_outputs = dict()
        # thermal_img_tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        # rgb_img_tensor = torch.zeros((batch_size, *batch[0][1].shape), dtype=torch.uint8)
        thermal_img_tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        rgb_img_tensor = torch.zeros((batch_size, *batch[0][1].shape), dtype=torch.float32)

        # Weights tensors
        rgb_weights = torch.zeros(batch_size, dtype=torch.float32)
        thermal_weights = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            thermal_img_tensor[i] += torch.from_numpy(batch[i][0])
            rgb_img_tensor[i] += torch.from_numpy(batch[i][1])
            labeler_inputs = {}
            for tk, tv in batch[i][2].items():
                instance_info = self.instance_info.get(tk, None)
                if instance_info is not None:
                    # target tensor is associated with a detection instance
                    tv = torch.from_numpy(tv).to(dtype=torch.float32)
                    if self.anchor_labeler is None:
                        if i == 0:
                            shape = (batch_size,) + instance_info['shape']
                            target_tensor = torch.full(shape, instance_info['fill'], dtype=torch.float32)
                            target[tk] = target_tensor
                        else:
                            target_tensor = target[tk]
                        num_elem = min(tv.shape[0], self.max_instances)
                        target_tensor[i, 0:num_elem] = tv[0:num_elem]
                    else:
                        # no need to pass gt tensors through when labeler in use
                        if tk in ('bbox', 'cls'):
                            labeler_inputs[tk] = tv
                else:
                    # target tensor is an image-level annotation / metadata
                    if i == 0:
                        # first batch elem, create destination tensors
                        if isinstance(tv, (tuple, list)):
                            # per batch elem sequence
                            shape = (batch_size, len(tv))
                            dtype = torch.float32 if isinstance(tv[0], (float, np.floating)) else torch.int32
                        else:
                            # per batch elem scalar
                            shape = batch_size,
                            dtype = torch.float32 if isinstance(tv, (float, np.floating)) else torch.int64
                        target_tensor = torch.zeros(shape, dtype=dtype)
                        target[tk] = target_tensor
                    else:
                        target_tensor = target[tk]
                    target_tensor[i] = torch.tensor(tv, dtype=target_tensor.dtype)

            if self.anchor_labeler is not None:
                cls_targets, box_targets, num_positives = self.anchor_labeler.label_anchors(
                    labeler_inputs['bbox'], labeler_inputs['cls'], filter_valid=False)
                if i == 0:
                    # first batch elem, create destination tensors, separate key per level
                    for j, (ct, bt) in enumerate(zip(cls_targets, box_targets)):
                        labeler_outputs[f'label_cls_{j}'] = torch.zeros(
                            (batch_size,) + ct.shape, dtype=torch.int64)
                        labeler_outputs[f'label_bbox_{j}'] = torch.zeros(
                            (batch_size,) + bt.shape, dtype=torch.float32)
                    labeler_outputs['label_num_positives'] = torch.zeros(batch_size)
                for j, (ct, bt) in enumerate(zip(cls_targets, box_targets)):
                    labeler_outputs[f'label_cls_{j}'][i] = ct
                    labeler_outputs[f'label_bbox_{j}'][i] = bt
                labeler_outputs['label_num_positives'][i] = num_positives
            
            # Add weights
            rgb_weights[i] = batch[i][3]  # rgb_weight
            thermal_weights[i] = batch[i][4]  # thermal_weight

        if labeler_outputs:
            target.update(labeler_outputs)

        return thermal_img_tensor, rgb_img_tensor, target, rgb_weights, thermal_weights


class PrefetchLoader:

    def __init__(self,
            loader,
            rgb_mean=IMAGENET_DEFAULT_MEAN,
            rgb_std=IMAGENET_DEFAULT_STD,
            thermal_mean=IMAGENET_DEFAULT_MEAN,
            thermal_std=IMAGENET_DEFAULT_STD,
            re_prob=0.,
            re_mode='pixel',
            re_count=1,
            ):
        self.loader = loader
        self.rgb_mean = torch.tensor([x * 255 for x in rgb_mean]).cuda().view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([x * 255 for x in rgb_std]).cuda().view(1, 3, 1, 1)
        self.thermal_mean = torch.tensor([x * 255 for x in thermal_mean]).cuda().view(1, 3, 1, 1)
        self.thermal_std = torch.tensor([x * 255 for x in thermal_std]).cuda().view(1, 3, 1, 1)

        if re_prob > 0.:
            self.random_erasing = RandomErasing(probability=re_prob, mode=re_mode, max_count=re_count)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True


        for next_thermal_input, next_rgb_input, next_target, next_rgb_weight, next_thermal_weight in self.loader:
            with torch.cuda.stream(stream):
                next_thermal_input = next_thermal_input.cuda(non_blocking=True)
                next_thermal_input = next_thermal_input.float().sub_(self.thermal_mean).div_(self.thermal_std)
                next_rgb_input = next_rgb_input.cuda(non_blocking=True)
                next_rgb_input = next_rgb_input.float().sub_(self.rgb_mean).div_(self.rgb_std)
                next_target = {k: v.cuda(non_blocking=True) for k, v in next_target.items()}

                if self.random_erasing is not None:
                    next_thermal_input, next_rgb_input = self.random_erasing(next_thermal_input, next_rgb_input, next_target)

            if not first:
                yield thermal_input, rgb_input, target, rgb_weight, thermal_weight
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            thermal_input = next_thermal_input
            rgb_input = next_rgb_input
            target = next_target
            rgb_weight = next_rgb_weight
            thermal_weight = next_thermal_weight

        yield thermal_input, rgb_input, target, rgb_weight, thermal_weight

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        re_prob=0.,
        re_mode='pixel',
        re_count=1,
        interpolation='bilinear',
        fill_color='mean',
        rgb_mean=IMAGENET_DEFAULT_MEAN,
        rgb_std=IMAGENET_DEFAULT_STD,
        thermal_mean=IMAGENET_DEFAULT_MEAN,
        thermal_std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        pin_mem=False,
        anchor_labeler=None,
        transform_fn=None,
        collate_fn=None
):
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if transform_fn is not None:
        # transform_fn should accept inputs (img, annotations) from the dataset and return a tuple
        # of img, annotations for the data loader collate function.
        # The valid types of img and annotations depend on the dataset and collate abstractions used.
        # The default dataset outputs PIL Image and dict of numpy ndarrays or python scalar annotations.
        # The fast collate fn accepts ONLY numpy uint8 images and annotations dicts of ndarrays and python scalars
        transform = transform_fn
    else:
        if is_training:
            transform = transforms_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                fill_color=fill_color,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                thermal_mean=thermal_mean,
                thermal_std=thermal_std)
        else:
            transform = transforms_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                fill_color=fill_color,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                thermal_mean=thermal_mean,
                thermal_std=thermal_std)
            
    dataset.transform = transform

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    collate_fn = collate_fn or DetectionFastCollate(anchor_labeler=anchor_labeler)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )
    if use_prefetcher:
        if is_training:
            loader = PrefetchLoader(loader, rgb_mean=rgb_mean, rgb_std=rgb_std, thermal_mean=thermal_mean, thermal_std=thermal_std, re_prob=re_prob, re_mode=re_mode, re_count=re_count)
        else:
            loader = PrefetchLoader(loader, rgb_mean=rgb_mean, rgb_std=rgb_std, thermal_mean=thermal_mean, thermal_std=thermal_std)

    return loader
