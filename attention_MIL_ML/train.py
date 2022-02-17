import torch
from torch import nn
from torch import optim
import numpy as np
from xiehe.attentionMIL import dataload
from torch.utils.data import DataLoader
from xiehe.attentionMIL import model
from torch.autograd import Variable
from sklearn import metrics as mt
import os
import warnings

warnings.filterwarnings("ignore")
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1, 2, 3"


def trains(model, train_loader, epoch, criterion, use_gpu):
    model.train()
    losss = 0
    for iter, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        if use_gpu:
            inputs = Variable(batch[0].cuda())
            labels = Variable(batch[1].cuda())
        else:
            inputs, labels = Variable(batch[0]), Variable(batch[1])

        optimizer.zero_grad()
        outputs, a = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        losss = losss + loss.item()
        if (iter + 1) % 10 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, iter, len(train_loader),
                    100. * iter / len(train_loader), losss / iter))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01 * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def va(gt2, pr2, th):
    value_0 = {'tp': 0, 'tn': 0, 'fn': 0, 'fp': 0}
    for i in range(len(gt2)):
        if gt2[i] == 1 and pr2[i] >= th:
            value_0['tp'] = value_0['tp'] + 1  # 真正例
        if gt2[i] == 0 and pr2[i] >= th:
            value_0['fp'] = value_0['fp'] + 1  # 假正例
        if gt2[i] == 0 and pr2[i] < th:
            value_0['tn'] = value_0['tn'] + 1  # 真负例
        if gt2[i] == 1 and pr2[i] < th:
            value_0['fn'] = value_0['fn'] + 1  # 假负例
    return value_0


def ODIR_Metrics_test(gt2, pr2, esp=0.000001):
    gt2 = gt2.flatten()[1::2]
    pr2 = pr2.flatten()[1::2]
    print(pr2)
    fpr, tpr, thresholds = mt.roc_curve(gt2, pr2)
    max = 0
    th = 0
    for i in range(len(thresholds)):
        value_0 = va(gt2, pr2, thresholds[i])
        specificity = value_0['tn'] / (value_0['tn'] + value_0['fp'] + esp)  # 特异度
        sentific = value_0['tp'] / (value_0['tp'] + value_0['fn'] + esp)  # 特异度
        G_mean = np.sqrt(specificity * sentific)
        if G_mean > max:
            max = G_mean
            th = thresholds[i]
    value_0 = va(gt2, pr2, th)
    acc = (value_0['tp'] + value_0['tn']) / (value_0['tp'] + value_0['tn'] + value_0['fp'] + value_0['fn'])  # 准确度

    return acc, mt.roc_auc_score(gt2, pr2)


def val_test(model, test_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        p = []
        g = []
        for iter, batch in enumerate(test_loader):
            torch.cuda.empty_cache()
            if use_gpu:
                inputs = Variable(batch[0].cuda())
                labels = Variable(batch[1].cuda())
            else:
                inputs, labels = Variable(batch[0]), Variable(batch[1])
            outputs, a = model(inputs)
            loss = criterion(outputs, labels)
            outputs = outputs.data.cpu().numpy()
            labels = labels.cpu().numpy()
            for x, y in zip(outputs, labels):
                p.append(x)
                g.append(y)
            val_loss += loss.item()
    acc, auc = ODIR_Metrics_test(np.array(g), np.array(p))

    return acc, auc


class WeightedMultilabel(torch.nn.Module):

    def __init__(self, weights: torch.Tensor):
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.weights = weights.unsqueeze()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets) * self.weights


if __name__ == "__main__":
    batch_size = 12
    batch_size_s = 12
    epochs = 500
    lr = 0.00001
    momentum = 0.95
    w_decay = 1e-6
    step_size = 50
    gamma = 0.5
    n_class = 2
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    print('Load Train and Test Set')
    train_path, test_path = dataload.data_depent()
    train = dataload.MyDataset(train_path, transform=True)
    test = dataload.MyDataset(test_path, transform=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    model_dir = "/xiehe/attention_MIL_ML/save_model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model = model.Attention()
    if use_gpu:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=num_gpu)

    # pos_weight = torch.FloatTensor([1, 1])
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = WeightedMultilabel(weights=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    score_dir = os.path.join(model_dir, 'scores')
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    best_loss = 0
    t_max_auc = 0
    for epoch in range(1, epochs):
        adjust_learning_rate(optimizer, epoch)

        trains(model, train_loader, epoch, criterion, use_gpu)
        acc, auc = val_test(model, test_loader)
        print('.....................', epoch, '.......................')
        print('test acc:.....................', acc, 'test auc:.......................', auc)

        if auc > t_max_auc:
            t_max_auc = auc
            model_path = os.path.join(model_dir,
                                      str(epoch) + "_" + str(acc) + "_" + str(auc) + '_test_resnet.pth')
            torch.save(model, model_path)
