class Config(object):
    img_path = "/Users/hulk/Documents/SCUT_FBP5500_Asian/images_fine/"#"./SCUT_FBP5500_Asian/images_fine/"
    train_txt = "./data/train.txt"
    val_txt = "./data/val.txt"
    test_txt = "./data/pipeline.txt"
    log_dir = './log'
    weight_dir = './weight'
    backbone = "resnet50"
    loss = "focal_loss"
    metric = "arc_margin"
    optimizer = 'sgd'
    lr = 1e-1  # initial learning rate
    lr_step = 30
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    batch_size = 16
    val_batch_size = 60
    test_batch_size = 1
    num_workers = 8
    epochs = 50
    log_steps = 100
    save_epoch = 10
    num_classes = 5
    easy_margin = False
    cuda = True,
    fp16 = False,
    opt_level = 'O1'
    fine_tune = True
    bias = 10
    pretrained_model = "./weight_local/beauty_ft_epoch_50.pth"

