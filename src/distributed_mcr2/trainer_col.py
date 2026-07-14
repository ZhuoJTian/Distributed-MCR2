import torch
import torch.nn as nn
import os
import time
from .dataset.mnist import infiniteloop
from .loss import MCRLoss, ColMCRLoss3, ColMCRLoss3_5
import torch.nn.functional as F
import numpy as np

class MCRTrainer():
    def __init__(self,
                 netD,
                 optD,
                 dataset,
                 batchsize,
                 num_steps,
                 client_id,
                 path,
                 lr_decay=None,
                 device=None,
                 num_class=10):

        super(MCRTrainer, self).__init__()
        self.netD = netD
        self.optD = optD
        self.lr_decay = lr_decay
        self.dataset = dataset
        self.batchsize = batchsize
        self.num_steps = num_steps
        self.client_id = client_id
        self.device = device
        self.path = path 
        self.mcr_loss = MCRLoss(eps=0.5, numclasses=num_class)

    def train(self):
        """
                Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        self.netD.train()
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = 1000, shuffle=True)
        self.netD.to(self.device)

        # while step < self.num_steps:
        for idx, (data, label) in enumerate(dataloader):
            # data, label = next(iter_dataloader)
            # Format batch and label
            real_cpu = data.to(self.device)
            
            # print(real_cpu.shape)
            real_label = label.detach().to(self.device)
            
            self.netD.zero_grad()
            self.optD.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            err, item1, item2 = self.mcr_loss(Z, real_label)
            loss = err.item()
            loss_item1 = item1.item()
            loss_item2 = item2.item()

            err.backward()
            self.optD.step()
            
            # print("client_id:", self.client_id, ", epoch:", epoch, ", step:", step, ", lossD:", lossD, ", lossG:", lossG)
            
        return loss, loss_item1, loss_item2
    

class ColMCRTrainer3():
    def __init__(self,
                 netD,
                 optD,
                 dataset,
                 batchsize,
                 num_steps,
                 client_id,
                 path,
                 lr_decay=None,
                 device=None,
                 num_class=10,
                 num_nei=0,
                 rho = 0.01):

        super(ColMCRTrainer3, self).__init__()
        self.netD = netD
        self.optD = optD
        self.lr_decay = lr_decay
        self.dataset = dataset
        self.batchsize = batchsize
        self.num_steps = num_steps
        self.client_id = client_id
        self.device = device
        self.path = path 
        self.num_class = num_class
        self.num_nei = num_nei
        self.rho = rho
        self.colmcr_loss = ColMCRLoss3(gamma1=1.0, gamma2=1.0, eps=0.5, rho=rho, numclasses=num_class, num_neig=num_nei)

    def train(self, V_old, V_neig, Y_old):
        """
                Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        # update Y  k*d x d matrix, same dimension as V_old and V_neig[j]
        self.netD.eval()
        Y = Y_old + 0.1*sum([(V_old - V_neig[j]) for j in range(self.num_nei)])

        self.netD.train()
        self.netD.to(self.device)

        # while step < self.num_steps:
        # one epoch of local updating for Z_i
        for epoch in range(1):
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = 1000, shuffle=True)
            for idx, (data, label) in enumerate(dataloader):
                # data, label = next(iter_dataloader)
                # Format batch and label
                real_cpu = data.to(self.device)

                # print(real_cpu.shape)
                real_label = label.detach().to(self.device)
                
                self.netD.zero_grad()
                self.optD.zero_grad()

                # Forward pass real batch through D
                Z = self.netD(real_cpu)
                err, term1, term2, other_term1, other_term2 = self.colmcr_loss(Z, real_label, V_old, V_neig, Y)
                loss = err.item()
                loss_term1 = term1.item()
                loss_term2 = term2.item()

                err.backward()
                self.optD.step()
                
                # print("client_id:", self.client_id, ", epoch:", epoch, ", step:", step, ", lossD:", lossD, ", lossG:", lossG)
        
        return loss, loss_term1, loss_term2, other_term1, other_term2, Y

    
class ColMCRTrainer3_5():
    def __init__(self,
                 netD,
                 optD,
                 dataset,
                 batchsize,
                 num_steps,
                 client_id,
                 path,
                 total_receinodes_perclass,
                 Si,
                 lr_decay=None,
                 device=None,
                 num_class=10,
                 nei=0,
                 rho = 0.01):

        super(ColMCRTrainer3_5, self).__init__()
        self.netD = netD
        self.optD = optD
        self.lr_decay = lr_decay
        self.dataset = dataset
        self.batchsize = batchsize
        self.num_steps = num_steps
        self.client_id = client_id
        self.device = device
        self.path = path 
        self.num_class = num_class
        self.nei = nei
        self.rho = rho
        self.colmcr_loss = ColMCRLoss3_5(gamma1=1.0, gamma2=1.0, eps=0.5, rho=rho, \
                                       numclasses=num_class, neig=nei, total_receinodes_perclass=total_receinodes_perclass, si=Si)
        

    def train(self, label_s, V_cluster, num_V_cluster, V_old, V_neig, Y):
        """
                Runs the training pipeline with all given parameters in Trainer.
        """
        self.netD.train()
        self.netD.to(self.device)
        # while step < self.num_steps:
        # one epoch of local updating for Z_i
        num_epoch = 3
        for epoch in range(num_epoch):
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = 3000, shuffle=True)
            for idx, (data, label) in enumerate(dataloader):
                # data, label = next(iter_dataloader)
                # Format batch and label
                real_cpu = data.to(self.device)

                # print(real_cpu.shape)
                real_label = label.detach().to(self.device)
                
                self.netD.zero_grad()
                self.optD.zero_grad()

                # Forward pass real batch through D
                Z = self.netD(real_cpu)
                err, term1, term2, other_term1, other_term2 = self.colmcr_loss(Z, real_label, label_s, V_cluster, num_V_cluster, V_old, V_neig, Y)
                loss = err.item()
                loss_term1 = term1.item()
                loss_term2 = term2.item()

                err.backward()
                self.optD.step()
                
                # print("client_id:", self.client_id, ", epoch:", epoch, ", step:", step, ", lossD:", lossD, ", lossG:", lossG)
        
        return loss, loss_term1, loss_term2, other_term1, other_term2
  