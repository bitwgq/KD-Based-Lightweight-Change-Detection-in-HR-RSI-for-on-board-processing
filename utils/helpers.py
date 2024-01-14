import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from utils.dataloaders import (full_path_loader, full_test_loader, CDDloader)
from utils.metrics import jaccard_loss, dice_loss
from utils.losses import hybrid_loss
from models.siamunet.siamunet_conc import SiamUNet_conc
from models.FC_Siam.model import FC_Siam
from models.DSIFN.DSIFN import DSIFN
logging.basicConfig(level=logging.INFO)

def initialize_metrics():

    metrics = {
        'loss_ce': 0,
        'loss_kd': 0,
        'cd_corrects': 0,
        'cd_precisions': 0,
        'cd_recalls': 0,
        'cd_f1scores': 0,
        'cd_kappa': 0,
        'learning_rate': 0,
    }

    return metrics


def get_mean_metrics(metric_dict):

    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, loss_ce, loss_kd, acc, pre, rec, f1, kappa, lr):

    metric_dict['loss_ce'] = loss_ce.item()
    metric_dict['loss_kd'] = loss_kd.item()
    metric_dict['cd_corrects'] = acc
    metric_dict['cd_precisions'] = pre
    metric_dict['cd_recalls'] = rec
    metric_dict['cd_f1scores'] = f1
    metric_dict['cd_kappa'] = kappa
    metric_dict['learning_rate'] = lr

    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    test_dataset = CDDloader(test_full_load, aug=False)

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_loader


def get_criterion(opt):

    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'ce':
        criterion = nn.CrossEntropyLoss()
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion


def load_model(opt, device):

    if opt.model_t == 'r50':
        model_t = FC_Siam(
            encoder_name="resnet50",
            encoder_weights='imagenet',
            in_channels=3,
            classes=2,
            siam_encoder=True,
            fusion_form='concat',
        )
    elif opt.model_t == 'vgg19':
        model_t = FC_Siam(
            encoder_name="vgg19_bn",
            encoder_weights='imagenet',
            in_channels=3,
            classes=2,
            siam_encoder=True,
            fusion_form='concat',
        )
    elif opt.model_t == 'IFN':
        model_t = DSIFN()

    if opt.model_s == 'base':
        model_s = SiamUNet_conc(in_ch=opt.num_channel, out_ch=2)


    model_t = model_t.to(device)
    model_s = model_s.to(device)

    return model_s, model_t
