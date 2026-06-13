from __future__ import absolute_import
from __future__ import print_function

from client import ColMCRClient3
import torch
import copy
import numpy as np
import os
# import datasets_old
from model import Encoder_CIFAR, Classifier
from args import parse_args
from general_utils import setup_seed, plot_corZ
from Sample_parti import getAttr
from dataset import cifar10
import torch.nn.functional as F

def create_clients(datasets_train, datasets_test, num_classes, dim_z, adj, path):
    clients = []
    for k in range(len(datasets_train)):
        neig_location = list(np.nonzero(adj[:, k])[0])
        num_neig = len(neig_location)
        client = ColMCRClient3(client_id=k,
                        data_set_train=datasets_train[k],
                        data_set_test=datasets_test[k],
                        num_classes=num_classes,
                        dimz=dim_z,
                        neig=neig_location,
                        num_neig=num_neig,
                        device="cuda",
                        path = path)
        clients.append(client)
    return clients

def set_up_clients(args, adj, models):
    lCCategoryToClients, LocalDist, LocaDis_test, SamplesToClients, SamplesToClients_test, MAX_NUM = getAttr(args.num_clients)
    choices = [[0, 1] for i in range(args.num_clients)]
    local_datasets_train, local_datasets_train_all, local_datasets_test = \
        cifar10.create_datasets(args.data_dir, args.dataset,
                              args.num_clients, choices, LocalDist, LocaDis_test)
    # plot_datasets(args.num_clients, local_datasets_train)
    print("load the datasets")
    # Ini_model = RestNet18_att()
    clients = create_clients(local_datasets_train, local_datasets_test, 
                             args.num_classes, args.dim_z, adj, args.path_result)
    for client_id in range(args.num_clients):
        client = clients[client_id]
        model = models[client_id]
        encoder = Encoder_CIFAR(args.dim_z, model)
        classifier= Classifier(args.num_classes, args.dim_z)
        # initialize model parameters
        client.netD = copy.deepcopy(encoder).to("cpu")
        client.netC = copy.deepcopy(classifier).to("cpu")
        del(encoder, classifier)
        opt_D = torch.optim.Adam(client.netD.parameters(), lr=0.01, weight_decay=args.weight_decay)
        opt_C = torch.optim.Adam(client.netC.parameters(), lr=0.01, weight_decay=args.weight_decay)
        # initialize T and Y
        client.Y = torch.zeros((args.dim_z*args.num_classes, args.dim_z))
        client.getV()
        client.setup(batch_size=args.batch_size,
                     num_local_epochs=1,
                     optimizer_D=opt_D,
                     optimizer_C=opt_C
                     )
        del opt_D, opt_C
    clients = get_Vneig(clients, args)
    print("load the clients and initialized the parameters")
    return clients

def get_Vneig(clients, args):
    for client_id in range(args.num_clients):
        client = clients[client_id]
        V_neig = [0] * client.num_neig
        for j in range(client.num_neig):
            V_neig[j] = clients[client.neig[j]].V_old
        client.V_neig = V_neig
    return clients

def Agg_model(clients, adj_matrix, Models):
    weight_keys = list(Models[0].keys())
    for client_id in range(len(clients)):
        client = clients[client_id]
        model_dict = Models[client_id]
        client_vars_sum = copy.deepcopy(model_dict)
        neig_location = list(np.nonzero(adj_matrix[:, client_id])[0])
        num_neig = len(neig_location)
        # select output channel
        for k in weight_keys:
            temp = client_vars_sum[k]
            for i in neig_location:
                temp += Models[i][k]
            client_vars_sum[k] = torch.true_divide(temp, num_neig+1)
        client.netC.load_state_dict(client_vars_sum)
    return clients

def main():
    adj = np.loadtxt("./10_adj_matrix.txt")
    args = parse_args()
    setup_seed(args.seed)
    args.num_clients = 10
    args.num_classes = 10
    args.path_result = "./Exp_results/IID/CIFAR10/ColMCR/"
    args.epochs = 1000
    models = ['res18', 'res34', 'vgg11', 'vgg16', 'res18', 'res34', 'vgg11', 'vgg16', 'res18', 'res34']
    clients = set_up_clients(args, adj, models)
    print("client set up")
    path_model = args.path_result+"models/"
    os.makedirs(path_model, exist_ok=True)
    # Start training

    for epoch in range(args.epochs):
        train_loss_all = []
        test_loss_all = []
        Z_all = []
        label_all = []
        agent_all = []
        for client_id in range(args.num_clients): # tqdm(range(args.num_clients), ascii=True):  
            client = clients[client_id]
            train_loss, loss_term1, loss_term2, other_term1, other_term2 = client.client_pretrain()
            test_loss = client.client_pretest()
            client.getV()
            train_loss_all.append(train_loss)
            test_loss_all.append(test_loss)

            if (epoch+1)%1==0:
                # client.client_savemodel(path_model)
                file_path = args.path_result + "loss_" + str(client_id) + ".txt"
                with open(file_path, "a+") as f:
                    f.write("epoch {} ".format(epoch+1))
                    f.write("Trainloss: {:.4f}, Testloss: {:.4f}, Trainloss_term1: {:.4f}, Trainloss_term2: {:.4f}, Trainloss_other1: {:.4f}, Trainloss_other2: {:.4f}".format(\
                        train_loss, test_loss, loss_term1, loss_term2, other_term1, other_term2))
                    f.write("\n")
        
            if (epoch+1)%100==0:
                client.client_savemodel(path_model)
                Z_list, label_list = client.getz_all_list()
                Z_all += copy.deepcopy(Z_list)                
                label_all += copy.deepcopy(label_list)
                agent_all += copy.deepcopy([client_id for i in range(len(Z_list))])
                del Z_list, label_list

        clients = get_Vneig(clients, args)

        if (epoch+1)%1==0:
            file_path = args.path_result + "averaged_result" + ".txt"
            with open(file_path, "a+") as f:
                f.write("epoch {} ".format(epoch+1))
                f.write("Trainloss: {:.4f}, Testloss: {:.4f}".format(np.average(train_loss_all), np.average(test_loss_all)))
                f.write("\n")

        
        if (epoch+1)%100 == 0:
            path_fig = args.path_result + "CorZ_" + str(epoch+1) + ".png"
            plot_corZ(Z_all, label_all, path_fig)

        # evaluate on test set
    print("end of pretraining")


if __name__ == "__main__":
    main()
