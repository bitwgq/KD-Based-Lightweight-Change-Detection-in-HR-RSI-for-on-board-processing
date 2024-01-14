import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import (get_test_loaders)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = 'xxx/xxx.pt'   # the path of the model
model = torch.load(path)
model = model.to(dev)

CM_test = 0
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)

        _, cd_preds = torch.max(cd_preds, 1)

        CM_NEW = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                  cd_preds.data.cpu().numpy().flatten())
        if CM_NEW.size == 1:
            CM_NEW = np.atleast_2d(CM_NEW)
            CM_NEW = np.pad(CM_NEW, (0, 1), 'constant', constant_values=(0))
        CM_test = CM_test + CM_NEW

tp_sum = CM_test[1, 1]
pred_sum = tp_sum + CM_test[0, 1]
true_sum = tp_sum + CM_test[1, 0]
tp_all = tp_sum + CM_test[0, 0]
all = CM_test[0, 1] + CM_test[1, 0] + tp_all
acc = tp_all / all
pre = tp_sum / pred_sum
rec = tp_sum / true_sum
f1 = 2 * pre * rec / (pre+rec)
po = acc
pe1 = ((CM_test[0, 0]+ CM_test[0, 1])*(CM_test[0, 0]+ CM_test[1, 0]) + (CM_test[1, 0]+ CM_test[1, 1])*(CM_test[0, 1]+ CM_test[1, 1]))
pe2 = all*all
pe = pe1 / pe2
kappa = (po - pe) / (1 - pe)

print('Precision: {}\nRecall: {}\nF1-Score: {}\nOA: {}\nKappa: {}'.format(pre, rec, f1, acc, kappa))