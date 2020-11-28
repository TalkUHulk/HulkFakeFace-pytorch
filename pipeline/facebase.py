import abc
from tool import timer
import torch
from collections import OrderedDict

class FaceBase(abc.ABC):

    def __init__(self, weight_path, cuda):
        self.weight_path = weight_path
        self.model = None
        self.initialized = False
        self._cuda = cuda

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__

    def initialize(self, **kwargs):
        checkpoint = torch.load(self.weight_path, map_location='cuda' if self._cuda else 'cpu')
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                name = k.lstrip('module').lstrip('.')
                new_checkpoint[name] = v
            else:
                new_checkpoint[k] = v

        self.model.load_state_dict(new_checkpoint)
        if self._cuda:
            self.model.cuda()
        self.model.eval()
        self.initialized = True

    def run(self, *args, **kwargs):
        """
        an abstract method need to be implemented
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized.")
        return self.inference(*args, **kwargs)

    @timer
    @torch.no_grad()
    def inference(self, *args, **kwargs):
        pass
