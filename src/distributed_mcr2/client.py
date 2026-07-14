import numpy as np
import torch
from .trainer_col import MCRTrainer, ColMCRTrainer3, ColMCRTrainer3_5  

import torch.nn.functional as F
from .general_utils import accuracy_softmax, accuracy_topk
from .loss import MCRLoss, ColMCRLoss3, ColMCRLoss3_5 

class CEClient(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, dimz, neig, num_neig, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.device = device
        self.neig = neig
        self.dimz = dimz
        self.num_neig = num_neig
        self.neig_z = 0
        self.neig_z_label = 0
        self.__netD = None
        self.__netC = None
        self.path = path

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]

    def client_train(self):
        """Update local model using local dataset."""
        self.netD.train()
        self.netC.train()
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)
        self.netD.to(self.device)
        self.netC.to(self.device)

        # while step < self.num_steps:
        for idx, (data, label) in enumerate(dataloader):
            # data, label = next(iter_dataloader)
            # Format batch and label
            real_cpu = data.to(self.device)
            real_label = label.clone().detach().to(self.device)
            
            self.netD.zero_grad()
            self.optD.zero_grad()
            self.netC.zero_grad()
            self.optC.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            output = self.netC(Z)
            err = F.cross_entropy(output, real_label)
            loss = err.item()
            err.backward()
            self.optD.step()
            self.optC.step()

        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)):
                real_cpu = data.to(self.device)
                feature = self.netD.to(self.device)(real_cpu)
                output = self.netC.to(self.device)(feature)
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        acc = 1.0*correct/len(self.dataset_train)
        return loss, acc

    def client_test(self):
        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_test, batch_size = 300, shuffle=True)): 
                feature = self.netD.to(self.device)(data.to(self.device))
                output = self.netC.to(self.device)(feature.to(self.device))
                loss_cross = F.cross_entropy(output.to(self.device), label.to(self.device))
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        acc = 1.0*correct/len(self.dataset_test)

        if self.device == "cuda": torch.cuda.empty_cache()
        return loss_cross.item(), acc

    def getz_all_list(self):
        self.netD.eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = int(len(self.dataset_train)/4), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD.to(self.device)(real_cpu)
                Z_list = Z.tolist()
                label_list = label.tolist()
                Z_all += Z_list
                label_all += label_list
        del dataloader, data, real_cpu
        return Z_all, label_all

    def client_savemodel(self, path):
        torch.save(self.netD.state_dict(), path + str(self.id) + "_netD.pt")
        torch.save(self.netC.state_dict(), path + str(self.id) + "_netC.pt")

class MCRClient(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.device = device
        self.__netD = None
        self.__netC = None
        self.path = path

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]
        self.pretrainer = MCRTrainer(
            netD=self.netD,
            optD=self.optD,
            dataset = self.dataset_train,
            batchsize = self.batchsize,
            path = self.path,
            num_steps=1,  # T
            client_id=self.id,
            lr_decay='linear',
            device=self.device,
            num_class=self.num_classes
        )

    def client_pretrain(self):
        """Update local model using local dataset."""
        loss, loss_item1, loss_item2 = self.pretrainer.train()
        self.netD.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        return loss, loss_item1, loss_item2

    def client_pretest(self):
        loss = []
        test_loss = 0
        self.netD.eval()
        with torch.no_grad():
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size = len(self.dataset_test), shuffle=True)
            for idx2, (data_test, label_test) in enumerate(dataloader_test):
                feature_test = self.netD.to("cpu")(data_test)
                mcr_loss = MCRLoss(eps=0.5, numclasses=self.num_classes)
                err, _, _ = mcr_loss(feature_test, label_test)
                loss.append(err.detach().item())
        self.netD.to("cpu")
        self.netC.to("cpu")

        if self.device == "cuda": torch.cuda.empty_cache()
        test_loss = np.average(loss)
        return test_loss
    
    def client_clsftrain(self):
        self.netD.eval()
        self.netC.train()
        self.netD.to(self.device)
        self.netC.to(self.device)
        correct = 0
        # while step < self.num_steps:
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)
        for idx, (data, label) in enumerate(dataloader):
            real_cpu = data.to(self.device)
            real_label = label.clone().detach().to(self.device)
            
            self.netC.zero_grad()
            self.optC.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            output = self.netC(Z)
            loss_cross = F.cross_entropy(output.to(self.device), real_label.to(self.device))

            loss_cross.backward()   
            self.optC.step()
            
        self.netC.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_train, batch_size = len(self.dataset_train), shuffle=True)):
                real_cpu = data.to(self.device)
                feature = self.netD.to(self.device)(real_cpu)
                output = self.netC.to(self.device)(feature)
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        acc = correct/len(self.dataset_train)
        return loss_cross.item(), acc

    def client_clsftest(self):
        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_test, batch_size = 300, shuffle=True)): 
                feature = self.netD.to(self.device)(data.to(self.device))
                output = self.netC.to(self.device)(feature.to(self.device))
                loss_cross = F.cross_entropy(output.to(self.device), label.to(self.device))
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        acc = 1.0*correct/len(self.dataset_test)
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        return loss_cross.item(), acc
    
    def getz_all_list(self):
        self.netD.eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = int(len(self.dataset_train)/4), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD.to(self.device)(real_cpu)
                Z_list = Z.tolist()
                label_list = label.tolist()
                Z_all += Z_list
                label_all += label_list
        del dataloader, data, real_cpu
        return Z_all, label_all

    def client_savemodel(self, path):
        torch.save(self.netD.state_dict(), path + str(self.id) + "_netD.pt")
        torch.save(self.netC.state_dict(), path + str(self.id) + "_netC.pt")

class ColMCRClient3(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, dimz, neig, num_neig, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.dimz = dimz
        self.device = device
        self.neig = neig
        self.rho = 5.0
        self.num_neig = num_neig
        self.__netD = None
        self.__netC = None
        self.path = path
        self.V_old = torch.tensor(0)
        self.V_neig = torch.tensor(0)
        self.Y = torch.tensor(0)

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]
        self.pretrainer = ColMCRTrainer3(
            netD=self.netD,
            optD=self.optD,
            dataset = self.dataset_train,
            batchsize = self.batchsize,
            path = self.path,
            num_steps=1,  # T
            client_id=self.id,
            lr_decay='linear',
            device=self.device,
            num_class=self.num_classes,
            num_nei=self.num_neig,
            rho=self.rho
        )

    def client_pretrain(self):
        """Update local model using local dataset."""
        loss, loss_term1, loss_term2, other_term1, other_term2, Y  = self.pretrainer.train(\
            self.V_old.to(self.device), [item.to(self.device) for item in self.V_neig], self.Y.to(self.device))
        self.netD.to("cpu")
        self.Y = Y.detach().clone().to("cpu")
        self.V_old.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        return loss, loss_term1, loss_term2, other_term1, other_term2

    def client_pretest(self):
        test_loss = 0
        self.netD.eval()
        with torch.no_grad():
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size = int(len(self.dataset_test)/5), shuffle=True)
            for idx2, (data_test, label_test) in enumerate(dataloader_test):
                feature_test = self.netD.to(self.device)(data_test.to(self.device))
                colmcr_loss = ColMCRLoss3(gamma1 = 1.0, gamma2 = 1.0, eps=0.5, rho=self.rho, numclasses=self.num_classes, num_neig=self.num_neig)
                err, _, _, _, _ = colmcr_loss(feature_test, label_test, \
                                           self.V_old.to(self.device), [item.to(self.device) for item in self.V_neig], self.Y.to(self.device))
                test_loss += err.item()*data_test.shape[0]
        self.netD.to("cpu")
        self.Y.to("cpu")
        self.V_old.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        test_loss = test_loss/len(self.dataset_test)
        return test_loss
    
    def client_clsftrain(self):
        self.netD.eval()
        self.netC.train()
        self.netD.to(self.device)
        self.netC.to(self.device)
        correct = 0
        # while step < self.num_steps:
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)
        for idx, (data, label) in enumerate(dataloader):
            real_cpu = data.to(self.device)
            real_label = label.clone().detach().to(self.device)
            
            self.netC.zero_grad()
            self.optC.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            output = self.netC(Z)
            loss_cross = F.cross_entropy(output.to(self.device), real_label.to(self.device))

            loss_cross.backward()   
            self.optC.step()
        self.netC.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_train, batch_size = len(self.dataset_train), shuffle=True)):
                real_cpu = data.to(self.device)
                feature = self.netD.to(self.device)(real_cpu)
                output = self.netC.to(self.device)(feature)
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        acc = correct/len(self.dataset_train)
        return loss_cross.item(), acc

    def client_clsftest(self):
        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_test, batch_size = 300, shuffle=True)): 
                feature = self.netD.to(self.device)(data.to(self.device))
                output = self.netC.to(self.device)(feature.to(self.device))
                loss_cross = F.cross_entropy(output.to(self.device), label.to(self.device))
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        acc = 1.0*correct/len(self.dataset_test)
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        return loss_cross.item(), acc

    def getV(self):
        self.netD.eval()
        self.netC.eval()
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = len(self.dataset_train), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):    
                real_cpu = data.to(self.device)
                Z = self.netD.to(self.device)(real_cpu)
                Pi = F.one_hot(label, self.num_classes)
                V = torch.zeros((self.dimz*self.num_classes, self.dimz))
                for k in range(self.num_classes):
                    Z_ = Z[Pi[:, k] == 1, :]
                    if Z_.shape[0]!=0:
                        V[k*self.dimz:(k+1)*self.dimz, :] = Z_.T@Z_/Z_.shape[0] 
        self.V_old = V.detach().clone().to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()

    def getz_all_list(self):
        self.netD.eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = int(len(self.dataset_train)/4), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD.to(self.device)(real_cpu)
                Z_list = Z.tolist()
                label_list = label.tolist()
                Z_all += Z_list
                label_all += label_list
        del dataloader, data, real_cpu
        return Z_all, label_all

    def client_savemodel(self, path):
        torch.save(self.Y, path + str(self.id) + "_Y.pt")
        torch.save({'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optD.state_dict(),
                    }, path + str(self.id) + "_netD.pt")
        torch.save({'model_state_dict': self.netC.state_dict(),
                    'optimizer_state_dict': self.optC.state_dict(),
                    }, path + str(self.id) + "_netC.pt")


class ColMCRClient3_5(object):
    def __init__(self, client_id, data_set_train, data_set_test, num_classes, dimz, neig, num_neig, total_receinodes_perclass, Si, local_label, device, path):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.dataset_train = data_set_train
        self.dataset_test = data_set_test
        self.num_classes = num_classes
        self.dimz = dimz
        self.total_receinodes_perclass = total_receinodes_perclass
        self.device = device
        self.neig = neig
        self.rho = 2.0
          
        self.local_label = local_label
        self.Si=Si
        self.num = [0 for i in range(self.num_classes)]
        self.num_neig = num_neig
        self.__netD = None
        self.__netC = None
        self.path = path
        self.V_old = torch.tensor(0)
        self.VV = torch.tensor(0)
        self.V_neig = torch.tensor(0)
        self.Y = torch.tensor(0)


    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__netD, self.__netC

    @model.setter
    def model(self, netD, netC):
        """Local model setter for passing globally aggregated model parameters."""
        self.__netD = netD
        self.__netC = netC

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optD = client_config["optimizer_D"]
        self.optC = client_config["optimizer_C"]
        self.batchsize = client_config["batch_size"]
        self.pretrainer = ColMCRTrainer3_5(
            netD=self.netD,
            optD=self.optD,
            dataset = self.dataset_train,
            batchsize = self.batchsize,
            path = self.path,
            total_receinodes_perclass=self.total_receinodes_perclass,
            Si=self.Si,
            num_steps=1,  # T
            client_id=self.id,
            lr_decay='linear',
            device=self.device,
            num_class=self.num_classes,
            nei=self.neig,
            rho=self.rho
        )

    def client_pretrain(self, label_s, V_cluster, num_V_cluster):
        """Update local model using local dataset."""
        loss, loss_term1, loss_term2, other_term1, other_term2  = self.pretrainer.train(\
            label_s, [item.to(self.device) for item in V_cluster], num_V_cluster, self.V_old.to(self.device), [item.to(self.device) for item in self.V_neig], self.Y.to(self.device))        
        self.netD.to("cpu")
        self.V_old.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        return loss, loss_term1, loss_term2, other_term1, other_term2

    def client_pretest(self, V_cluster, num_V_cluster):
        test_loss = 0
        self.netD.eval()
        with torch.no_grad():
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size = int(len(self.dataset_test)/5), shuffle=True)
            for idx2, (data_test, label_test) in enumerate(dataloader_test):
                feature_test = self.netD.to(self.device)(data_test.to(self.device))
                colmcr_loss = ColMCRLoss3_5(gamma1 = 1.0, gamma2 = 1.0, eps=0.5, rho=self.rho, numclasses=self.num_classes, num_neig=self.num_neig)
                err, _, _, _, _ = colmcr_loss(feature_test, label_test, V_cluster, num_V_cluster,\
                                            self.V_old.to(self.device), [item.to(self.device) for item in self.V_neig], self.Y.to(self.device))
                                           
                test_loss += err.item()*data_test.shape[0]
        self.netD.to("cpu")
        self.Y.to("cpu")
        self.V_old.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        test_loss = test_loss/len(self.dataset_test)
        return test_loss

    def client_clsftrain(self):
        self.netD.eval()
        self.netC.train()
        self.netD.to(self.device)
        self.netC.to(self.device)
        correct = 0
        # while step < self.num_steps:
        dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = self.batchsize, shuffle=True)
        for idx, (data, label) in enumerate(dataloader):
            real_cpu = data.to(self.device)
            real_label = label.clone().detach().to(self.device)
            
            self.netC.zero_grad()
            self.optC.zero_grad()

            # Forward pass real batch through D
            Z = self.netD(real_cpu)
            output = self.netC(Z)
            loss_cross = F.cross_entropy(output.to(self.device), real_label.to(self.device))

            loss_cross.backward()   
            self.optC.step()
            break
        self.netC.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_train, batch_size = len(self.dataset_train), shuffle=True)):
                real_cpu = data.to(self.device)
                feature = self.netD.to(self.device)(real_cpu)
                output = self.netC.to(self.device)(feature)
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        acc = correct/len(self.dataset_train)
        return loss_cross.item(), acc

    def client_clsftest(self):
        self.netD.eval()
        self.netC.eval()
        correct = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(torch.utils.data.DataLoader(self.dataset_test, batch_size = 300, shuffle=True)): 
                feature = self.netD.to(self.device)(data.to(self.device))
                output = self.netC.to(self.device)(feature.to(self.device))
                loss_cross = F.cross_entropy(output.to(self.device), label.to(self.device))
                correct += accuracy_topk(F.softmax(output).to("cpu"), label.to("cpu"))[0]
        acc = 1.0*correct/len(self.dataset_test)
        self.netD.to("cpu")
        self.netC.to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
        return loss_cross.item(), acc

    def getV(self):
        self.netD.eval()
        self.netC.eval()
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = len(self.dataset_train), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):    
                real_cpu = data.to(self.device)
                Z = self.netD.to(self.device)(real_cpu)
                Pi = F.one_hot(label, self.num_classes)
                V = torch.zeros((self.dimz*self.num_classes, self.dimz))
                VV = torch.zeros((self.dimz*self.num_classes, self.dimz))
                for k in range(self.num_classes):
                    Z_ = Z[Pi[:, k] == 1, :]
                    self.num[k] = Z_.shape[0]/self.Si
                    if Z_.shape[0]!=0:
                        V[k*self.dimz:(k+1)*self.dimz, :] = Z_.T@Z_/Z_.shape[0] 
                        VV[k*self.dimz:(k+1)*self.dimz, :] = Z_.T@Z_/self.Si
        self.V_old = V.detach().clone().to("cpu")
        self.VV = VV.detach().clone().to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()
    
    def getVV_part(self, label_s):
        self.netD.eval()
        self.netC.eval()
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = len(self.dataset_train), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):    
                real_cpu = data.to(self.device)
                Z = self.netD.to(self.device)(real_cpu)
                Pi = F.one_hot(label, self.num_classes)
                for k in label_s:
                    Z_ = Z[Pi[:, k] == 1, :]
                    if Z_.shape[0]!=0:
                        self.VV[k*self.dimz:(k+1)*self.dimz, :] = Z_.T@Z_/self.Si
        self.VV.detach().to("cpu")
        if self.device == "cuda": torch.cuda.empty_cache()

    def getz_all_list(self):
        self.netD.eval()
        self.netC.eval()
        Z_all = []
        label_all = []
        with torch.no_grad():
            dataloader = torch.utils.data.DataLoader(self.dataset_train, batch_size = int(len(self.dataset_train)/4), shuffle=False)
            for idx, (data, label) in enumerate(dataloader):
                real_cpu = data.to(self.device)
                Z = self.netD.to(self.device)(real_cpu)
                Z_list = Z.tolist()
                label_list = label.tolist()
                Z_all += Z_list
                label_all += label_list
        del dataloader, data, real_cpu
        return Z_all, label_all

    def client_savemodel(self, path):
        torch.save(self.Y, path + str(self.id) + "_Y.pt")
        torch.save({'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optD.state_dict(),
                    }, path + str(self.id) + "_netD.pt")
        torch.save({'model_state_dict': self.netC.state_dict(),
                    'optimizer_state_dict': self.optC.state_dict(),
                    }, path + str(self.id) + "_netC.pt")
