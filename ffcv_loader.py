import numpy as np
import gin

from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage,RandomHorizontalFlip, View, Convert
from ffcv.transforms.color_jitter import RandomColorJitter
from ffcv.transforms.solarization import RandomSolarization
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder

import torch
import torchvision.transforms.v2 as tfms
from torch import nn


from multiprocessing import cpu_count
from typing import Any, Callable, Literal, Mapping, Sequence, Type, Union
import numpy as np
import torch
from ffcv.fields.base import Field
from ffcv.loader import Loader
from ffcv.loader.loader import (DEFAULT_PROCESS_CACHE, ORDER_MAP, ORDER_TYPE,
                                OrderOption)
from ffcv.memory_managers import (MemoryManager, OSCacheManager,
                                  ProcessCacheManager)
from ffcv.pipeline import Compiler, Pipeline, PipelineSpec
from ffcv.pipeline.graph import Graph
from ffcv.pipeline.operation import Operation
from ffcv.reader import Reader
from ffcv.traversal_order.base import TraversalOrder


class MultiLoader(Loader):
    def __init__(self,
                fname: str,
                batch_size: int,
                num_workers: int = -1,
                cache_type: int = DEFAULT_PROCESS_CACHE,
                order: Union[ORDER_TYPE, TraversalOrder] = OrderOption.SEQUENTIAL,
                distributed: bool = False,
                seed: int = None,  # For ordering of samples
                indices: Sequence[int] = None,  # For subset selection
                pipelines: Mapping[str,
                                Sequence[Union[Operation, torch.nn.Module]]] = {},
                custom_fields: Mapping[str, Type[Field]] = {},
                drop_last: bool = True,
                batches_ahead: int = 3,
                recompile: bool = False,  # Recompile at every epoch
                ):

        if distributed and order == OrderOption.RANDOM and (seed is None):
            print('Warning: no ordering seed was specified with distributed=True. '
                    'Setting seed to 0 to match PyTorch distributed sampler.')
            seed = 0
        elif seed is None:
            tinfo = np.iinfo('int32')
            seed = np.random.randint(0, tinfo.max)

        # We store the original user arguments to be able to pass it to the
        # filtered version of the datasets
        self._args = {
            'fname': fname,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'os_cache': cache_type,
            'order': order,
            'distributed': distributed,
            'seed': seed,
            'indices': indices,
            'pipelines': pipelines,
            'drop_last': drop_last,
            'batches_ahead': batches_ahead,
            'recompile': recompile
        }
        self.fname: str = fname
        self.batch_size: int = batch_size
        self.batches_ahead = batches_ahead
        self.seed: int = seed
        self.reader: Reader = Reader(self.fname, custom_fields)
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last
        self.distributed: bool = distributed
        self.code = None
        self.recompile = recompile

        if self.num_workers < 1:
            self.num_workers = cpu_count()

        Compiler.set_num_threads(self.num_workers)

        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        
        if cache_type == 0:
            self.memory_manager: MemoryManager = OSCacheManager(self.reader)
        elif cache_type == 1:
            self.memory_manager: MemoryManager = ProcessCacheManager(
                self.reader)
        elif cache_type == 2:
            from ffcv.memory_managers.shared_cache import SharedMemoryManager
            self.memory_manager: MemoryManager = SharedMemoryManager(self.reader)
        else:
            raise ValueError("Unknown cache type. Use 0 for process cache, 1 for os cache, or 2 for no cache.")
        
        if order in ORDER_MAP:
            self.traversal_order: TraversalOrder = ORDER_MAP[order](self)
        elif isinstance(order, TraversalOrder):
            self.traversal_order: TraversalOrder = order(self)
        elif issubclass(order, TraversalOrder):
            self.traversal_order: TraversalOrder = order(self)
        else:
            raise ValueError(f"Order {order} is not a supported order type or a subclass of TraversalOrder")

        memory_read = self.memory_manager.compile_reader()
        self.next_epoch: int = 0

        self.pipelines = {}
        self.pipeline_specs = {}
        self.field_name_to_f_ix = {}
        custom_pipeline_specs = {}

        # Creating PipelineSpec objects from the pipeline dict passed
        # by the user
        for output_name, spec in pipelines.items():
            if isinstance(spec, PipelineSpec):
                pass
            elif isinstance(spec, Sequence):
                spec = PipelineSpec(output_name, decoder=None, transforms=spec)
            elif spec is None:
                continue  # This is a disabled field
            else:
                msg  = f"The pipeline for {output_name} has to be "
                msg += f"either a PipelineSpec or a sequence of operations"
                raise ValueError(msg)
            custom_pipeline_specs[output_name] = spec

        # Adding the default pipelines
        default_name_to_f_ix={}
        for f_ix, (field_name, field) in enumerate(self.reader.handlers.items()):
            default_name_to_f_ix[field_name] = f_ix

        # We add the custom fields after the default ones
        # This is to preserve backwards compatibility and make sure the order
        # is intuitive
        for field_name, spec in custom_pipeline_specs.items():
            # redirect
            self.field_name_to_f_ix[field_name] = default_name_to_f_ix[spec.source]

            if field_name not in self.pipeline_specs:
                self.pipeline_specs[field_name] = spec

        self.graph = Graph(self.pipeline_specs, self.reader.handlers,
                            self.field_name_to_f_ix, self.reader.metadata,
                            memory_read)
        
        self.generate_code()
        self.first_traversal_order = self.next_traversal_order()
        
        
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

@gin.configurable
def SimplePipeline(img_size=224,scale=(0.2,1), ratio=(3.0/4.0, 4.0/3.0),device='cuda'):
    device = torch.device(device)
    image_pipeline = [
            RandomResizedCropRGBImageDecoder((img_size, img_size), scale=scale,ratio=ratio,),
            RandomHorizontalFlip(),          
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),  
            ToTensor(),  ToTorchImage(),
            ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device), View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines    

@gin.configurable
def ValPipeline(img_size=224,ratio= 224/256,device='cuda'):
    device = torch.device(device)
    image_pipeline = [
            CenterCropRGBImageDecoder((img_size, img_size), ratio),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ToTensor(),  ToTorchImage(),
            ToDevice(device),
            ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device), View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines  

from torchvision.transforms import InterpolationMode

class ThreeAugmentation(nn.Module):
    """Apply single transformation randomly picked from a list. This transform does not support torchscript."""

    def __init__(self, ):
        super().__init__()
        self.guassian_blur = tfms.GaussianBlur(3,sigma=(0.1,2))
        self.solarize = tfms.RandomSolarize(0,1)
        self.grayscale = tfms.RandomGrayscale(p=1)

    def __call__(self, x):
        op_index = torch.randint(0,3,(len(x),))
        for i,op in enumerate([self.guassian_blur,
                               self.solarize,
                               self.grayscale]):
            tf_mask = op_index == i
            x[tf_mask] = op(x[tf_mask])
        return x

    def extra_repr(self) -> str:
        return "GaussianBlur, Solarize, Grayscale"
        
@gin.configurable
def ThreeAugmentPipeline(img_size=224,scale=(0.08,1), color_jitter=None,device='cuda'):
    """
    ThreeAugmentPipeline: https://github.com/facebookresearch/deit/blob/main/augment.py
    """
    if not color_jitter is None: assert color_jitter >= 0 and color_jitter <= 1
    device = torch.device(device)
    image_pipeline = (
        # first_tfl 
        [   RandomResizedCropRGBImageDecoder((img_size, img_size), scale=scale,),
            RandomHorizontalFlip(),]+
        # second_tfl
        (   [RandomColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter,hue=0, p=0.5)] if color_jitter else []) + 
        # final_tfl
        [
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ToTensor(), ToTorchImage(),
            ToDevice(device),
            ThreeAugmentation(),
        ]) 
        
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device),View(-1)]
    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    } 
    return pipelines   

@gin.configurable
def ColorJitterPipeline(img_size=224,scale=(0.08, 1.0),device='cuda'):
    device = torch.device(device)
    image_pipeline = [
        RandomHorizontalFlip(),
        RandomColorJitter(0.8, 0.4, 0.4, 0.2, p=0.1),
        RandomSolarization(128,p=0.2),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ToTensor(), ToTorchImage(),
        ToDevice(device,non_blocking=True),
        tfms.RandomGrayscale(p=0.1),
        tfms.GaussianBlur(3, sigma=(0.1, 2)),
    ]
    label_pipeline = [IntDecoder(), ToTensor(),ToDevice(device),View(-1)]
    # Pipeline for each data field
    from ffcv.pipeline import PipelineSpec
    pipelines = {
        'image': PipelineSpec("image",RandomResizedCropRGBImageDecoder((img_size, img_size),scale=scale),transforms=image_pipeline),             
    } 
    pipelines['label'] = label_pipeline
    return pipelines

@gin.configurable
def MultiviewPipeline(img_size=224,scale=(0.4, 1.0),local_crops_number=0,
                      local_img_size=96,device='cuda'):
    k = local_img_size/img_size
    local_scale=(scale[0]*k, scale[1]*k)
    
    image_pipeline = [
        RandomHorizontalFlip(),
        RandomColorJitter(0.8, 0.4, 0.4, 0.2, p=0.1),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ToTensor(), ToTorchImage(),
        ToDevice(torch.device(device),non_blocking=True),
        tfms.RandomGrayscale(p=0.1),
        tfms.GaussianBlur(3, sigma=(0.1, 2)),
    ]
    image_pipeline2 = [
        RandomHorizontalFlip(),
        RandomColorJitter(0.8, 0.4, 0.4, 0.2, p=0.1),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ToTensor(), ToTorchImage(),
        ToDevice(torch.device(device),non_blocking=True),
        tfms.RandomGrayscale(p=0.1),
        tfms.GaussianBlur(3, sigma=(0.1, 2)),
        tfms.RandomSolarize(0,0.2), # asymmetric augmentation
    ]
    def _local_pipeline():
        return [
            RandomHorizontalFlip(),
            RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ToTensor(), ToTorchImage(),
            Convert(torch.float16),
            ToDevice(torch.device(device),non_blocking=True),
        ]
    label_pipeline = [IntDecoder(), ToTensor(),View(-1)]
    # Pipeline for each data field
    from ffcv.pipeline import PipelineSpec
    pipelines = {
        'image': PipelineSpec("image",RandomResizedCropRGBImageDecoder((img_size, img_size),scale=scale),transforms=image_pipeline),
        'image2': PipelineSpec("image",RandomResizedCropRGBImageDecoder((img_size, img_size),scale=scale),transforms=image_pipeline2),        
    } 
    for i in range(local_crops_number):
        pipelines[f"local_{i}"] = PipelineSpec("image",RandomResizedCropRGBImageDecoder((local_img_size, local_img_size),scale=local_scale),transforms=_local_pipeline())
    pipelines['label'] = label_pipeline
    return pipelines

from ffcv.loader import Loader, OrderOption
from ffcv.traversal_order.base import TraversalOrder
from torch.utils.data import DistributedSampler

def get_values(loader, field_name:str) -> 'Loader':
    new_args = {**loader._args}
    pipelines = {}
    old_pipelines = loader._args['pipelines']
    # Disabling all the other fields
    for other_field_name in loader.reader.handlers.keys():
        pipelines[other_field_name] = None

    # We reuse the original pipeline for the field we care about
    try:
        pipelines[field_name] = old_pipelines[field_name]
    except KeyError:
        # We keep the default one if the user didn't setup a custom one
        del pipelines[field_name]
        pass

    new_args['pipelines'] = pipelines

    # We use sequential order for speed and to know which index we are
    # filtering
    new_args['order'] = OrderOption.SEQUENTIAL
    new_args['drop_last'] = False
    new_args['distributed'] = False
    sub_loader = Loader(**new_args)
    values = []
    # Iterate through the loader and test the user defined condition
    for batch, in sub_loader:
        values.append(batch.clone())
    return torch.cat(values)

class ArbitraryView(TraversalOrder):

    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        self.labels = get_values(loader,'label').cpu().numpy().astype(np.uint64)
        
        self.label_indices = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}
        if self.distributed:
            self.sampler = DistributedSampler(self.indices,
                                              shuffle=True,
                                              seed=self.seed,
                                              drop_last=False)
        # assert loader.batch_size % 4 == 0, "batch size must be divisible by 4"
        
                
    def sample_order(self, epoch: int) :
        if not self.distributed:
            generator = np.random.default_rng(self.seed + epoch if self.seed is not None else None)
            order_list = generator.permutation(self.indices)
        else:
            self.sampler.set_epoch(epoch)

            order_list = self.indices[np.array(list(self.sampler))]

        # create a new order list with positive samples
        instance_order = order_list.copy()
        
        # find the order of the class        
        class_order = np.zeros_like(instance_order)
        labels = self.labels[instance_order]
        for y in np.unique(self.labels):
            y_indices = self.label_indices[y]
            np.random.shuffle(y_indices)
            y_mask = labels==y
            class_order[y_mask] = y_indices[:y_mask.sum()]
                    
        # arbitrary order
        arbitrary_order = np.random.permutation(order_list) # inverse the order
        ## solve the conflict
        old_class = self.labels[order_list]
        conflict_mask = old_class == self.labels[arbitrary_order]
        sub_old_class = old_class[conflict_mask]
        sub_new_class = np.random.randint(0, len(self.label_indices)-1, size=(conflict_mask.sum()))
        sub_new_class[sub_new_class >= sub_old_class] += 1 # skipp the old class
        
        ## overwrite the class order
        ref_conflict = arbitrary_order[conflict_mask].copy()
        
        for y in np.unique(sub_new_class):
            y_mask = sub_new_class == y
            y_indices = self.label_indices[y]
            
            ref_conflict[y_mask] = np.random.choice(y_indices, y_mask.sum())
            
        arbitrary_order[conflict_mask] = ref_conflict
                   
        return np.stack([order_list, instance_order, class_order, arbitrary_order], axis=1).flatten()