from .client import ColMCRClient3_5
import torch
import copy
import numpy as np

# import datasets_old
from .model import Encoder_CIFAR, Classifier
from .args import parse_args
from .general_utils import setup_seed, plot_corZ
from .sample_parti_noiid import getAttr
from .dataset import cifar10
from .paths import resolve_input_path, resolve_output_path
from pathlib import Path

def create_clients(datasets_train, datasets_test, LocalDist, CategoryToClients, num_classes, dim_z, total_receinodes, total_receinodes_perclass, adj, path, device):
    clients = []
    for k in range(len(datasets_train)):
        neig_location = total_receinodes[k]
        num_neig = len(neig_location)
        Si=1
        client = ColMCRClient3_5(client_id=k,
                        data_set_train=datasets_train[k],
                        data_set_test=datasets_test[k],
                        num_classes=num_classes,
                        dimz=dim_z,
                        neig=neig_location,
                        num_neig=num_neig,
                        total_receinodes_perclass = total_receinodes_perclass[k],
                        local_label=CategoryToClients[k],
                        Si=Si,
                        device=device,
                        path=path)
        clients.append(client)
    return clients

def set_up_clients(args, models, adj):
    CategoryToClients, Categories_client, LocalDist, LocaDis_test, SamplesToClients, SamplesToClients_test, MAX_NUM = getAttr(args.num_clients, 4)
    choices = [[0, 1] for i in range(args.num_clients)]
    local_datasets_train, local_datasets_train_all, local_datasets_test = \
        cifar10.create_datasets(args.data_dir, args.dataset,
                              args.num_clients, choices, LocalDist, LocaDis_test)
    # plot_datasets(args.num_clients, local_datasets_train)
    print("load the datasets")
    # print(CategoryToClients, Categories_client)
    # Ini_model = RestNet18_att()
    total_receinodes, total_receinodes_perclass = get_receivenodes(args.num_clients, CategoryToClients, Categories_client)
    # print(total_receinodes)
    clients = create_clients(local_datasets_train, local_datasets_test, LocalDist, CategoryToClients,
                             args.num_classes, args.dim_z, total_receinodes, total_receinodes_perclass, adj, args.path_result, args.device)
    for client_id in range(args.num_clients):
        client = clients[client_id]
        model = models[client_id]
        encoder = Encoder_CIFAR(args.dim_z, model)
        classifier= Classifier(args.num_classes, args.dim_z)
        # initialize model parameters
        
        client.netD = copy.deepcopy(encoder).to("cpu")
        client.netC = copy.deepcopy(classifier).to("cpu")
        opt_D = torch.optim.Adam(client.netD.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        opt_C = torch.optim.Adam(client.netC.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        '''
        path_model = str(Path(args.path_result) / "models")
    Path(path_model).mkdir(parents=True, exist_ok=True)+ str(client_id)
        checkpoint = torch.load(path_model + "_netD.pt")
        client.netD.load_state_dict(checkpoint['model_state_dict'])
        client.netD.eval()
        
        opt_D.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in opt_D.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        checkpoint = torch.load(path_model + "_netC.pt")
        client.netC.load_state_dict(checkpoint['model_state_dict'])
        client.netC.eval()
        
        opt_C.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in opt_C.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        '''
        del(encoder, classifier)
        # initialize T and Y
        # client.Y = torch.load(path_model + "_Y.pt")
        client.Y = torch.zeros((args.dim_z*args.num_classes, args.dim_z))
        client.getV()
        client.setup(batch_size=args.batch_size,
                     num_local_epochs=args.num_local_epochs,
                     optimizer_D=opt_D,
                     optimizer_C=opt_C
                     )
        del opt_D, opt_C
    clients = get_Vneig(clients, args)
    print("load the clients and initialized the parameters")
    return clients

def get_receivenodes(num_clients, CategoryToClients, Categories_client):
    total_receinodes = []
    total_receinodes_perclass = []
    for i in range(num_clients):
        receinodes = []
        recei_perclass = [[] for ii in range(10)]
        for classi in CategoryToClients[i]:
            receinodes += [j for j in Categories_client[classi] if (j!=i)]
            recei_perclass[classi] = [j for j in Categories_client[classi] if (j!=i)]
        total_receinodes.append(list(set(receinodes)))
        total_receinodes_perclass.append(recei_perclass)
    return total_receinodes, total_receinodes_perclass

def get_Vneig(clients, args):
    for client_id in range(args.num_clients):
        client = clients[client_id]
        V_neig = [0] * client.num_neig
        for j in range(client.num_neig):
            V_neig[j] = clients[client.neig[j]].V_old
        client.V_neig = V_neig
    return clients


def get_V_cluster(clients, args, receive_idx, Cluster, Labels):
    VVKnodes = [torch.zeros((args.dim_z, args.dim_z)) for i in range(args.num_classes)]
    VV_num = [0 for i in range(args.num_classes)]
    for idx in receive_idx:
        node = Cluster[idx]
        client = clients[node]
        for label in Labels[idx]:
            VVKnodes[label] += client.VV[label*args.dim_z:(label+1)*args.dim_z, :]
            VV_num[label] += client.num[label]
    # print(VVKnodes, VV_num)
    return VVKnodes, VV_num

def _resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def _load_adjacency(path, num_clients):
    matrix_path = resolve_input_path(path)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Adjacency matrix not found: {matrix_path}")
    adjacency = np.loadtxt(matrix_path)
    if adjacency.shape != (num_clients, num_clients):
        raise ValueError(
            f"Expected a {num_clients}x{num_clients} adjacency matrix, "
            f"but {matrix_path.name} has shape {adjacency.shape}."
        )
    return adjacency


def main():
    args = parse_args(
        default_num_clients=4,
        default_result_path="Exp_results/NIID/CIFAR10_S/ColMCR",
        default_adjacency_file="4_adj_matrix.txt",
    )
    args.device = _resolve_device(args.device)
    args.data_dir = str(resolve_output_path(args.data_dir))
    args.path_result = str(resolve_output_path(args.path_result))
    Path(args.path_result).mkdir(parents=True, exist_ok=True)
    adj = _load_adjacency(args.adjacency_file, args.num_clients)
    setup_seed(args.seed)
    models = ['res18', 'res18', 'res18', 'res18']# ['res18', 'vgg11', 'vgg16', 'res34', 'vgg11']
    clients = set_up_clients(args, models, adj)
    print("client set up")
    path_model = str(Path(args.path_result) / "models")
    Path(path_model).mkdir(parents=True, exist_ok=True)
    # Start training
    Cluster = [
        [0, 1], [2, 3]
    ]
    Labels1 = [[1,2,3,4,5], [0,6,7,8,9]]
    Labels2 = [[0,3,5,7,9], [1,2,4,6,8]]

    d=128
    for epoch in range(0, args.epochs):
        train_loss_all = []
        test_loss_all = []
        Z_all = []
        label_all = []
        agent_all = []
        for client_id in range(args.num_clients): # tqdm(range(args.num_clients), ascii=True):  
            client = clients[client_id]
            # update Y  dk x d matrix, same dimension as V_old and V_neig[j]
            for k in client.local_label:
                neig = client.total_receinodes_perclass[k]
                neig_idx = [client.neig.index(ii) for ii in neig]
                client.Y[k*d:(k+1)*d, :] = client.Y[k*d:(k+1)*d, :] + 0.1*sum([(client.V_old[k*d:(k+1)*d, :] - client.V_neig[j][k*d:(k+1)*d, :]) for j in neig_idx])

        for i in range(len(Cluster[0])):   
            client_id = Cluster[0][i]
            client = clients[client_id]
            label_s = Labels1[i]
            receive_idx = [j for j in range(len(Cluster[0])) if j!=i]
            VVKnodes, VV_num = get_V_cluster(clients, args, receive_idx, Cluster[0], Labels1)
            train_loss, loss_term1, loss_term2, other_term1, other_term2 = \
                client.client_pretrain(label_s, VVKnodes, VV_num)
            client.getVV_part(label_s)
            if (epoch+1)%1==0:
                # client.client_savemodel(path_model)
                file_path = Path(args.path_result) / f"loss_{client_id}.txt"
                with open(file_path, "a+") as f:
                    f.write("epoch {} ".format(epoch+1))
                    f.write("Trainloss: {:.4f}, Trainloss_term1: {:.4f}, Trainloss_term2: {:.4f}, Trainloss_other1: {:.4f}, Trainloss_other2: {:.4f}".format(\
                        train_loss, loss_term1, loss_term2, other_term1, other_term2))
                    f.write("\n")
        
        for i in range(len(Cluster[1])):   
            client_id = Cluster[1][i]
            client = clients[client_id]
            label_s = Labels2[i]
            receive_idx = [j for j in range(len(Cluster[1])) if j!=i]
            VVKnodes, VV_num = get_V_cluster(clients, args, receive_idx, Cluster[1], Labels2)
            train_loss, loss_term1, loss_term2, other_term1, other_term2 = \
                client.client_pretrain(label_s, VVKnodes, VV_num)
            client.getVV_part(label_s)
            if (epoch+1)%1==0:
                # client.client_savemodel(path_model)
                file_path = Path(args.path_result) / f"loss_{client_id}.txt"
                with open(file_path, "a+") as f:
                    f.write("epoch {} ".format(epoch+1))
                    f.write("Trainloss: {:.4f}, Trainloss_term1: {:.4f}, Trainloss_term2: {:.4f}, Trainloss_other1: {:.4f}, Trainloss_other2: {:.4f}".format(\
                        train_loss, loss_term1, loss_term2, other_term1, other_term2))
                    f.write("\n")

        for client_id in range(args.num_clients):
            client = clients[client_id]
            client.getV()
            if (epoch + 1) % args.save_every == 0:
                client.client_savemodel(path_model)
                Z_list, label_list = client.getz_all_list()
                Z_all += copy.deepcopy(Z_list)                
                label_all += copy.deepcopy(label_list)
                agent_all += copy.deepcopy([client_id for i in range(len(Z_list))])
                del Z_list, label_list 

        clients = get_Vneig(clients, args)

        if (epoch + 1) % args.save_every == 0:
            path_fig = str(Path(args.path_result) / f"CorZ_{epoch + 1}.jpg")
            plot_corZ(Z_all, label_all, agent_all, path_fig)

        # evaluate on test set
    print("end of pretraining")

if __name__ == "__main__":
    main()
