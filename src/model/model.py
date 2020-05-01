from __future__ import division
import tqdm
import torch
import numpy as np
# from optimizer import optim_schedule as OS
from torch.optim import Adam

class ModelTrainer(object):

    def __init__(self, model, train_data_loader, cfg, validation_data_loader=None): 
        self.model = model
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.cfg = cfg
        if torch.cuda.is_available() and cfg.has_key('gpuid'):
            self.device = torch.device('cuda:{}'.format(self.cfg.gpuid))
        else:
            self.device = torch.device('cpu')

        
        # if torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     cuda_devices = range(torch.cuda.device_count())
        #     self.model = torch.nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=0.01)
        #self.optim_schedule = OS.ScheduledOptim(self.optim, 100, n_warmup_steps=10000)


        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss()

            
    def train(self, epoch):
        self.iteration(epoch, self.train_data_loader)

    def predict(self, epoch=-1):
        self.iteration(epoch, self.validation_data_loader, train=False)


    def predict_assignment(self, p):
        
        data_point_device = {key: value.to(self.device) for key, value in p.items()
                              if type(value) == torch.Tensor}
        data_point_device.update({key:[v.to(self.device) for v in value] for key, value in p.items()
                           if type(value) == list})
        p = data_point_device
        
        nl = p["num_left_chunks"].to(self.device)
        nr = p["num_right_chunks"].to(self.device)
        
        lc = {idx : [str(ele) for ele in e.cpu().detach().numpy().tolist()] for idx, e in enumerate(p["left_chunk_ids"])}
        rc = {idx: [str(ele) for ele in e.cpu().detach().numpy().tolist()] for idx, e in enumerate(p["right_chunk_ids"])}
        rc[-1] = ["0"]
        lc[-1] = ["0"]
        
        y_hat,_ = self.model.forward(p)
        y_hat_na = y_hat[:, -1]
        y_hat = y_hat[:, :nr]
        y_hat = torch.cat((y_hat, y_hat_na.unsqueeze(1)), 1)
        y_hat = y_hat[:nl, :nr +1 ]
        y_pred = torch.zeros((nl, nr + 1))
        y_pred[range(nl), y_hat.argmax(1)] = 1.

        predictions = y_pred.argmax(1).detach().numpy()
        nrc = len(p["right_chunk_ids"])
        nlc = len(p["left_chunk_ids"])
        
        predictions[predictions > (nrc - 1)] = -1
        missing_idx = set(range(nrc)) - set(np.unique(predictions))
        predictions_text = [" ".join(lc[idx]) + " <==> " + " ".join(rc[p]) for idx, p in enumerate(predictions)]

        for mx in missing_idx:
            predictions_text.append(" ".join(lc[-1]) + " <==> " + " ".join(rc[mx]))
        
        return y_pred, predictions_text


    def loss(self, log_prediction, log_prediction_na_right, label, label_na, label_na_right, nl, nr):
        log_prediction = log_prediction.double()
        log_prediction_na = log_prediction[:,-1]
        log_prediction  = log_prediction[:, :nr]

        label = label[:nl, :nr]
        label_na = label_na[:nl]

        log_prediction = torch.cat((log_prediction, log_prediction_na.unsqueeze(1).double()), 1)
        log_prediction = log_prediction[:nl, :nr + 1]
        label = torch.cat((label, label_na.unsqueeze(1).double()), 1)

        weight = torch.ones((nl, nr + 1)).double().to(self.device)
        weight[:, -1] = 0.05


        #ll_left = torch.abs(label_left - torch.exp(log_prediction_left))
        categorical_cross_entropy_left = - (weight * label * log_prediction).double()

        label_na_right = label_na_right[:nr]
        log_prediction_na_right = log_prediction_na_right[:nr]
        categorical_cross_entropy_right = - (0.05 * label_na_right * log_prediction_na_right).double()

        return torch.mean(categorical_cross_entropy_left).double() + torch.mean(categorical_cross_entropy_right).double()
        
    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_correct = 0
        total_points = 0
        precision = 0.
        recall = 0.
        f1 = 0.
        total_loss = 0.
        
        for idx, batch in data_iter:
            batch_loss = 0.
            batch_data = []

            for data_point in batch:
                data_point_device = {key: value.to(self.device) for key, value in data_point.items()
                              if type(value) == torch.Tensor}
                data_point_device.update({key:[v.to(self.device) for v in value] for key, value in data_point.items()
                                   if type(value) == list})

                batch_data.append(data_point_device)

            if train:
                for p in batch_data:
                    nl, nr = p["num_left_chunks"], p["num_right_chunks"]
                    y_hat, y_hat_right = self.model.forward(p)
                    y = p["aligned"]
                    y_na_left = p["is_aligned"]
                    y_na_right = (~(torch.sum(p["aligned"], 0) > 0)).float()
                    
                    batch_loss += self.loss(y_hat, y_hat_right, y, y_na_left, y_na_right, nl, nr)

            if train:
                #self.optim_schedule.zero_grad()
                self.optim.zero_grad()
                batch_loss.backward()
                #self.optim_schedule.step_and_update_lr()
                self.optim.step()

            total_loss += batch_loss
            for p in batch_data:
                nl = p["num_left_chunks"]
                nr = p["num_right_chunks"]
                y_hat,_ = self.model.forward(p)
                y_hat_na = y_hat[:, -1]
                y_hat = y_hat[:, :nr]

                y_hat = torch.cat((y_hat, y_hat_na.unsqueeze(1)), 1)
                y_hat = y_hat[:nl, :nr +1 ]
                y_pred = torch.zeros((nl, nr + 1)).to(self.device)

                y_pred[range(nl), y_hat.argmax(1)] = 1.
                y_pred = y_pred.double()
                y = p["aligned"]
                y = y[:nl, :nr]
                y_na = p["is_aligned"]
                y_na = y_na[:nl]
                y = torch.cat((y, y_na.unsqueeze(1).double()), 1)
                

                y[y != 1.] = -1.
                y_pred[y_pred != 1.] = -2.

                correct_in_sample = y_pred.eq(y).sum().item()
                point_precision = float(correct_in_sample) / float(nr)
                point_recall = float(correct_in_sample) / float(nl)
                precision += point_precision
                recall += point_recall
                
                if point_precision + point_recall == 0.:
                    f1 += 0.
                else:
                    f1 += (2 * point_precision * point_recall) / (point_precision + point_recall)

                total_points += 1.

        print "f1", f1  / total_points, " precision", precision / total_points, " recall", recall  / total_points
        print "Total Loss", total_loss
            
            
            
