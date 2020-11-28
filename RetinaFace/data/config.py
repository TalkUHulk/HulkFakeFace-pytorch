# config.py

cfg_mnetv2 = {
    'name': 'mobilenet_v2',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'width_mult': 1,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 512,
    'pretrain': True,
    'return_layers': {'5': 1, '13': 2, '18': 3},
    'in_channel': [32, 96, 1280],
    'out_channel': 256,
    'gender': False
}


cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 12,
    'ngpu': 2,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': [512, 1024, 2048],
    'out_channel': 256,
    'gender': True
}
