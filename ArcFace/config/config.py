class Config(object):
    
    train_datasets = "../faces_glintasia/train_datasets/"
    lfw_root = '../validation/lfw/'
    lfw_test_list = '../validation/lfw_pair.txt'
    log_dir = './log'
    weight_dir = './weight'
    backbone = "resnet50"
    pretrained = "./weight/IRSeResNet50.pth"
    loss = "focal_loss"
    metric = "arc_margin"
    optimizer = 'sgd'
    lr = 1e-2  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    momentum = 0.9
    batch_size = 300
    test_batch_size = 60
    num_workers = 8
    epochs = 50
    log_steps = 100
    save_epoch = 10
    num_classes = 93979
    easy_margin = False
    cuda = True
    num_gpu = 1
    fp16 = False
    opt_level = 'O1'

    # ls -l |grep "^d"|wc -l
    # 18050    2 gpu
    # 192351  111
