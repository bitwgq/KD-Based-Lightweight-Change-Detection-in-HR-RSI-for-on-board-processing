import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.metrics import confusion_matrix
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion, get_test_loaders,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from criterion import CriterionPC
import warnings
warnings.filterwarnings("ignore")

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

logging.basicConfig(level=logging.INFO)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

train_loader, val_loader = get_loaders(opt)
test_loader = get_test_loaders(opt)

logging.info('LOADING Model')
model_s, model_t = load_model(opt, dev)

model_t = torch.load(opt.path_t)

criterion = get_criterion(opt)
klcri = nn.KLDivLoss()
l2loss = nn.MSELoss()
optimizer = torch.optim.SGD(model_s.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

cripc = CriterionPC(2)

logging.info('STARTING training')
total_step = -1

gamma = 5
alpha = 2
beta = 5e3
T1 = 2
T2 = 2

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()
    best_metrics = initialize_metrics()
    CM_train = 0
    CM_val = 0

    model_s.train()
    model_t.eval()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        optimizer.zero_grad()

        cd_preds_s, fea_s = model_s(batch_img1, batch_img2)
        with torch.no_grad():
            cd_preds_t, fea_t = model_t(batch_img1, batch_img2)
        loss_ce = criterion(cd_preds_s, labels)

        outputs_S = F.softmax(cd_preds_s.view(-1, 2, 256*256) / T1, dim=1)
        outputs_T = F.softmax(cd_preds_t.view(-1, 2, 256*256) / T1, dim=1)
        outputs_S2 = F.softmax(cd_preds_s.view(-1, 2, 256*256) / T2, dim=2)
        outputs_T2 = F.softmax(cd_preds_t.view(-1, 2, 256*256) / T2, dim=2)

        # CSN distillation
        loss_kd = alpha * klcri(outputs_S.log(), outputs_T) * T1 * T1 + beta * klcri(outputs_S2.log(), outputs_T2) * T2 * T2
        # PC distillation
        loss_fea = gamma * cripc(fea_s, fea_t, labels)
        loss = loss_ce + loss_kd + loss_fea
        loss.backward()
        optimizer.step()

        _, cd_preds = torch.max(cd_preds_s, 1)


        CM_NEW = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                  cd_preds.data.cpu().numpy().flatten())
        if CM_NEW.size == 1:
            CM_NEW = np.atleast_2d(CM_NEW)
            CM_NEW = np.pad(CM_NEW, (0, 1), 'constant', constant_values=(0))
        CM_train = CM_train + CM_NEW

        del batch_img1, batch_img2, labels

    print(loss_fea.item())

    tp_sum = CM_train[1, 1]
    pred_sum = tp_sum + CM_train[0, 1]
    true_sum = tp_sum + CM_train[1, 0]
    tp_all = tp_sum + CM_train[0, 0]
    all = CM_train[0, 1] + CM_train[1, 0] + tp_all
    acc = tp_all/all
    pre = tp_sum/pred_sum
    rec = tp_sum/true_sum
    f1 = 2*pre*rec/(pre+rec)
    po = acc
    pe1 = ((CM_train[0, 0]+ CM_train[0, 1])*(CM_train[0, 0]+ CM_train[1, 0]) + (CM_train[1, 0]+ CM_train[1, 1])*(CM_train[0, 1]+ CM_train[1, 1]))
    pe2 = all*all
    pe = pe1 / pe2
    kappa = (po - pe) / (1 - pe)
    train_metrics = set_metrics(train_metrics,
                                loss_ce,
                                loss_kd,
                                acc,
                                pre,
                                rec,
                                f1,
                                kappa,
                                scheduler.get_last_lr())
    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(train_metrics))

    model_s.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:

            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            cd_preds, fea_s = model_s(batch_img1, batch_img2)

            loss_ce = criterion(cd_preds, labels)

            _, cd_preds = torch.max(cd_preds, 1)

            CM_NEW = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                      cd_preds.data.cpu().numpy().flatten())
            if CM_NEW.size == 1:
                CM_NEW = np.atleast_2d(CM_NEW)
                CM_NEW = np.pad(CM_NEW, (0, 1), 'constant', constant_values=(0))
            CM_val = CM_val + CM_NEW

            del batch_img1, batch_img2, labels

        tp_sum = CM_val[1, 1]
        pred_sum = tp_sum + CM_val[0, 1]
        true_sum = tp_sum + CM_val[1, 0]
        tp_all = tp_sum + CM_val[0, 0]
        all = CM_val[0, 1] + CM_val[1, 0] + tp_all
        acc = tp_all / all
        pre = tp_sum / pred_sum
        rec = tp_sum / true_sum
        f1 = 2 * pre * rec / (pre + rec)
        po = acc
        pe1 = ((CM_val[0, 0] + CM_val[0, 1]) * (CM_val[0, 0] + CM_val[1, 0]) + (CM_val[1, 0] + CM_val[1, 1]) * (CM_val[0, 1] + CM_val[1, 1]))
        pe2 = all * all
        pe = pe1 / pe2
        kappa = (po - pe) / (1 - pe)
        val_metrics = set_metrics(val_metrics,
                                  loss_ce,
                                  loss_kd,
                                  acc,
                                  pre,
                                  rec,
                                  f1,
                                  kappa,
                                  scheduler.get_last_lr())
        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(val_metrics))

        if (val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']):
            logging.info('updata the model')
            metadata['validation_metrics'] = val_metrics
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            torch.save(model_s, './tmp/best'+'.pt')
            best_metrics = val_metrics

        print('An epoch finished.')

print('Train Done! The best result (val) ia:')
print(best_metrics)
