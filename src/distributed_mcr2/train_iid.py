from .client import ColMCRClient3
import torch
import copy
import numpy as np
from pathlib import Path
# import datasets_old
from .model import Encoder_CIFAR, Classifier
from .args import parse_args
from .general_utils import plot_corZ, setup_seed
from .paths import resolve_input_path, resolve_output_path
from .sample_parti import getAttr
from .dataset import cifar10

def create_clients(datasets_train, datasets_test, num_classes, dim_z, adj, path, device):
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
                        device=device,
                        path=path)
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
                             args.num_classes, args.dim_z, adj, args.path_result, args.device)
    for client_id in range(args.num_clients):
        client = clients[client_id]
        model = models[client_id]
        encoder = Encoder_CIFAR(args.dim_z, model)
        classifier= Classifier(args.num_classes, args.dim_z)
        # initialize model parameters
        client.netD = copy.deepcopy(encoder).to("cpu")
        client.netC = copy.deepcopy(classifier).to("cpu")
        del(encoder, classifier)
        opt_D = torch.optim.Adam(client.netD.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        opt_C = torch.optim.Adam(client.netC.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # initialize T and Y
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
    args = parse_args()
    args.device = _resolve_device(args.device)
    args.data_dir = str(resolve_output_path(args.data_dir))
    args.path_result = str(resolve_output_path(args.path_result))
    Path(args.path_result).mkdir(parents=True, exist_ok=True)
    adj = _load_adjacency(args.adjacency_file, args.num_clients)
    setup_seed(args.seed)
    models = ['res18', 'res34', 'vgg11', 'vgg16', 'res18', 'res34', 'vgg11', 'vgg16', 'res18', 'res34']
    clients = set_up_clients(args, adj, models)
    print("client set up")
    path_model = str(Path(args.path_result) / "models")
    Path(path_model).mkdir(parents=True, exist_ok=True)
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
                file_path = Path(args.path_result) / f"loss_{client_id}.txt"
                with open(file_path, "a+") as f:
                    f.write("epoch {} ".format(epoch+1))
                    f.write("Trainloss: {:.4f}, Testloss: {:.4f}, Trainloss_term1: {:.4f}, Trainloss_term2: {:.4f}, Trainloss_other1: {:.4f}, Trainloss_other2: {:.4f}".format(\
                        train_loss, test_loss, loss_term1, loss_term2, other_term1, other_term2))
                    f.write("\n")
        
            if (epoch + 1) % args.save_every == 0:
                client.client_savemodel(path_model)
                Z_list, label_list = client.getz_all_list()
                Z_all += copy.deepcopy(Z_list)                
                label_all += copy.deepcopy(label_list)
                agent_all += copy.deepcopy([client_id for i in range(len(Z_list))])
                del Z_list, label_list

        clients = get_Vneig(clients, args)

        if (epoch+1)%1==0:
            file_path = Path(args.path_result) / "averaged_result.txt"
            with open(file_path, "a+") as f:
                f.write("epoch {} ".format(epoch+1))
                f.write("Trainloss: {:.4f}, Testloss: {:.4f}".format(np.average(train_loss_all), np.average(test_loss_all)))
                f.write("\n")

        
        if (epoch + 1) % args.save_every == 0:
            path_fig = str(Path(args.path_result) / f"CorZ_{epoch + 1}.png")
            plot_corZ(Z_all, label_all, path_fig)

        # evaluate on test set
    print("end of pretraining")


if __name__ == "__main__":
    main()
