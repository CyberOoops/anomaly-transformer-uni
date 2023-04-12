import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def get_range_proba(predict, label, delay=7):
    predict = np.array(predict)
    label = np.array(label)

    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict

def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1 - predict) * (1 - actual))
    fp = np.sum(predict * (1 - actual))
    fn = np.sum((1 - predict) * actual)

    precision = (tp + 0.000001) / (tp + fp + 0.000001)
    recall = (tp + 0.000001) / (tp + fn + 0.000001)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall, tp, tn, fp, fn

def delay_f1(score, label, k=7):
    max_th = float(score.max())
    max_th = np.percentile(score, 99.91)
    print("max_th", max_th)
    min_th = float(score.min())
    grain = 2000
    max_f1_1 = 0.0
    max_f1_th_1 = 0.0
    pre = 0.0
    reca = 0.0
    for i in range(0, grain + 3):
        #print(i)
        thres = (max_th - min_th) / grain * i + min_th
        predict = score >= thres
        predict = get_range_proba(predict, label, k)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, label)
        if f1 > max_f1_1:
            max_f1_1 = f1
            max_f1_th_1 = thres
            pre = precision
            reca = recall
        predict = get_range_proba(score >= max_f1_th_1, label, k)
        #print(max_f1_th_1)
    return max_f1_1, pre, reca, predict

def point_adjust(score, label, thres):
    predict = score >= thres
    actual = label > 0.1
    anomaly_state = False

    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict, actual


def best_f1(score, label):
    # max_th = float(score.max())
    max_th = np.percentile(score, 99.91)
    print("max_th", max_th)
    min_th = float(score.min())
    grain = 2000
    max_f1_1 = 0.0
    max_f1_th_1 = 0.0
    pre = 0.0
    rec = 0.0
    for i in range(0, grain + 3):
        thres = (max_th - min_th) / grain * i + min_th
        predict, actual = point_adjust(score, label, thres=thres)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1_1:
            max_f1_1 = f1
            max_f1_th_1 = thres
            pre = precision
            rec = recall
    predict, actual = point_adjust(score, label, max_f1_th_1)
    print('thres',max_f1_th_1)
    return max_f1_1,pre,rec, predict

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        # attens_energy = []
        # for i, (input_data, labels) in enumerate(self.train_loader):
        #     input = input_data.float().to(self.device)
        #     output, series, prior, _ = self.model(input)
        #     loss = torch.mean(criterion(input, output), dim=-1)
        #     series_loss = 0.0
        #     prior_loss = 0.0
        #     for u in range(len(prior)):
        #         if u == 0:
        #             series_loss = my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss = my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature
        #         else:
        #             series_loss += my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss += my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature
        #     metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        #     cri = metric * loss
        #     cri = cri.detach().cpu().numpy()
        #     attens_energy.append(cri)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # train_energy = np.array(attens_energy)

        # # (2) find the threshold
        # attens_energy = []
        # for i, (input_data, labels) in enumerate(self.thre_loader):
        #     input = input_data.float().to(self.device)
        #     output, series, prior, _ = self.model(input)

        #     loss = torch.mean(criterion(input, output), dim=-1)

        #     series_loss = 0.0
        #     prior_loss = 0.0
        #     for u in range(len(prior)):
        #         if u == 0:
        #             series_loss = my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss = my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature
        #         else:
        #             series_loss += my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss += my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature
        #     # Metric
        #     metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        #     cri = metric * loss
        #     cri = cri.detach().cpu().numpy()
        #     attens_energy.append(cri)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # test_energy = np.array(attens_energy)
        # combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        # print("Threshold :", thresh)


        time1 = time.time()
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        time2 = time.time()

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)



        kk=7
        if self.dataset=='Yahoo':
            kk=3
        
        d_f1,d_pre,d_rec,predict = delay_f1(test_energy,test_labels,kk)
        m_f1,m_pre,m_rec,m_predict = best_f1(test_energy,test_labels)

        # pred = (test_energy > thresh).astype(int)

        # gt = test_labels.astype(int)

        # print("pred:   ", pred.shape)
        # print("gt:     ", gt.shape)

        # detection adjustment
        # anomaly_state = False
        # for i in range(len(gt)):
        #     if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
        #         anomaly_state = True
        #         for j in range(i, 0, -1):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #         for j in range(i, len(gt)):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #     elif gt[i] == 0:
        #         anomaly_state = False
        #     if anomaly_state:
        #         pred[i] = 1

        # kk=7
        # if self.dataset=='Yahoo':
        #     kk=3
        # new_pred = get_range_proba(pred,gt,kk)
        # pred = new_pred

        # pred = np.array(pred)
        # gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)

        # from sklearn.metrics import precision_recall_fscore_support
        # from sklearn.metrics import accuracy_score
        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
        #                                                                       average='binary')
        # print(
        #     "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        with open('./all_result.txt','a') as f:
            f.write('%f %f %f %f %f %f %f\n'%(time2-time1,d_f1,d_pre,d_rec,m_f1,m_pre,m_rec))
        print(d_f1,d_pre,d_rec)
        return 0, d_pre, d_rec, d_f1
