import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MCRLoss(nn.Module):

    def __init__(self, eps, numclasses):
        super(MCRLoss, self).__init__()

        self.num_class = numclasses
        self.faster_logdet = False
        self.eps = eps

    def forward(self, Z, real_label):
        err, item1, item2 = self.fast_version(Z, real_label)
        return err, item1, item2

    def fast_version(self, Z, real_label):
        """ decrease the times of calculate log-det  from 52 to 32"""
        z_total, (z_discrimn_item, z_compress_item, z_compress_losses, z_scalars) = self.deltaR(Z, real_label,
                                                                                                self.num_class)
        errD_mcr = z_discrimn_item - z_compress_item
        errD = -1.0 * errD_mcr 
        return errD, -1.0 * z_discrimn_item, z_compress_item

    def logdet(self, X):

        if self.faster_logdet:
            return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = self.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, Y, num_classes):

        Pi = F.one_hot(Y, num_classes).to(Z.device)
        discrimn_loss = self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term += s * z
        total_loss = discrimn_loss - compress_term

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)

class ColMCRLoss3(nn.Module):

    def __init__(self, gamma1, gamma2, eps, rho, numclasses, num_neig):
        super(ColMCRLoss3, self).__init__()

        self.num_class = numclasses
        self.faster_logdet = False
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eps = eps
        self.rho = rho
        self.num_neig = num_neig

    def forward(self, Z, real_label, V_old, V_neig, Y_old):
        err, term1, term2, other_term1, other_term2 = self.fast_version(Z, real_label, V_old, V_neig, Y_old)
        return err, term1, term2, other_term1, other_term2

    def fast_version(self, Z, real_label, V_old, V_neig, Y_old):

        """ decrease the times of calculate log-det  from 52 to 32"""

        z_total, (z_discrimn_item, z_compress_item, z_compress_losses, scalar) = \
                    self.deltaR(Z, real_label)
        other_term1, other_term2 = self.otherterms(Z, real_label, V_old, V_neig, Y_old)

        errD_mcr = self.gamma1 * z_discrimn_item - self.gamma2 * z_compress_item

        errD = -1.0 * errD_mcr + other_term1+ other_term2

        return errD, -1.0 * self.gamma1 * z_discrimn_item, self.gamma2 * z_compress_item, other_term1, other_term2

    def logdet(self, X):

        if self.faster_logdet:
            return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        scalar = d / (n * self.eps)
        logdet = self.logdet(I + scalar * Z @ Z.T)
        return logdet / 2.

    def compute_compress_loss(self, Z, Pi):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        for j in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum() + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * Z_ @ Z_.T)
            compress_loss.append(log_det)
            scalars.append(trPi / (2 * n))
        return compress_loss, scalars

    def deltaR(self, Z, label):
        Pi = F.one_hot(label, self.num_class).to(Z.device)
        n, d = Z.shape
        discrimn_loss =  self.compute_discrimn_loss(Z.T)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term +=  s * z
        total_loss = (discrimn_loss - compress_term)

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)
    
    def otherterms(self, Z, label, V_old, V_neig, Y):
        Pi = F.one_hot(label, self.num_class).to(Z.device)
        Z = Z.T
        d, n = Z.shape
        term1 = 0.
        term2 = 0.
        for k in range(Pi.shape[1]):
            Z_ = Z[:, Pi[:, k] == 1]
            term1 += 0. if Pi[:, k].sum() == 0 else sum([(Y[k*d:(k+1)*d, :].T @ ( Z_ @ Z_.T/Z_.shape[1]))[i, i] for i in range(d)]) 
            term2 += 0. if Pi[:, k].sum() == 0 else self.rho/2.0*sum([((torch.norm(Z_ @ Z_.T/Z_.shape[1] - \
                                                  1.0/2.0*(V_old[k*d:(k+1)*d, :] + V_neig[j][k*d:(k+1)*d, :]), p='fro'))**2) for j in range(self.num_neig)])
        return term1, term2


class ColMCRLoss3_5(nn.Module):

    def __init__(self, gamma1, gamma2, eps, rho, numclasses, neig, total_receinodes_perclass, si):
        super(ColMCRLoss3_5, self).__init__()

        self.num_class = numclasses 
        self.faster_logdet = False
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eps = eps
        self.rho = rho
        self.neig = neig
        self.total_receinodes_perclass = total_receinodes_perclass
        self.Si=si

    def forward(self, Z, real_label, label_s, V_cluster, num_V_cluster, V_old, V_neig, Y_old):

        """ decrease the times of calculate log-det  from 52 to 32"""

        z_total, (z_discrimn_item, z_compress_item, z_compress_losses, scalar) = \
                    self.deltaR(Z, real_label, label_s, V_cluster, num_V_cluster)
        other_term1, other_term2 = self.otherterms(Z, real_label, label_s, V_old, V_neig, Y_old)

        errD_mcr = self.gamma1 * z_discrimn_item - self.gamma2 * z_compress_item

        errD = -1.0 * errD_mcr + other_term1 + other_term2

        return errD, -1.0 * self.gamma1 * z_discrimn_item, self.gamma2 * z_compress_item, other_term1, other_term2

    def logdet(self, X):

        if self.faster_logdet:
            return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(X, upper=True))))
        else:
            return torch.logdet(X)

    def compute_discrimn_loss(self, Z, Pi, label_s, V_cluster, num_V_cluster):
        """Theoretical Discriminative Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        # print(label_s)
        V_cluster = [item.to(Z.device) for item in V_cluster]
        Z_ = torch.concatenate([Z[:, Pi[:, k] == 1] for k in label_s], 1)
        scalar = d / ((Z_.shape[1]/self.Si+sum(num_V_cluster)) * self.eps)
        term = Z_ @ Z_.T/self.Si + sum([V_cluster[j] for j in range(10) if j not in label_s])
        for j in label_s:
            if num_V_cluster[j]!=0:
                Z_ = Z[:, Pi[:, j] == 1]
                term += Z_ @Z_.T * num_V_cluster[j]/Z_.shape[1]
        logdet = self.logdet(I + scalar * term)
        return logdet / 2., Z_.shape[1]/self.Si+sum(num_V_cluster)

    def compute_compress_loss(self, Z, Pi, label_s, ms, V_cluster, num_V_cluster):
        """Theoretical Compressive Loss."""
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = []
        scalars = []
        V_cluster = [item.to(Z.device) for item in V_cluster]
        for j in label_s:
            Z_ = Z[:, Pi[:, j] == 1]
            trPi = Pi[:, j].sum()/self.Si + num_V_cluster[j] + 1e-8
            # trPi = Pi[:, j].sum()/self.Si  + 1e-8
            scalar = d / (trPi * self.eps)
            log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * (Z_ @ Z_.T/self.Si +  Z_ @ Z_.T * num_V_cluster[j]/Z_.shape[1]))
            # log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * (Z_ @ Z_.T/self.Si + V_cluster[j]))
            # log_det = 1. if Pi[:, j].sum() == 0 else self.logdet(I + scalar * (Z_ @ Z_.T/self.Si))
            compress_loss.append(1.0*log_det)
            scalars.append((Pi[:, j].sum()/self.Si + num_V_cluster[j]) / (2 * ms))
            # scalars.append((Pi[:, j].sum()/self.Si) / (2 * ms))
        return compress_loss, scalars

    def deltaR(self, Z, label, label_s, V_cluster, num_V_cluster):
        Pi = F.one_hot(label, self.num_class).to(Z.device)
        n, d = Z.shape
        discrimn_loss, ms =  self.compute_discrimn_loss(Z.T, Pi, label_s, V_cluster, num_V_cluster)
        compress_loss, scalars = self.compute_compress_loss(Z.T, Pi, label_s, ms, V_cluster, num_V_cluster)

        compress_term = 0.
        for z, s in zip(compress_loss, scalars):
            compress_term +=  s * z
        total_loss = (discrimn_loss - compress_term)

        return -total_loss, (discrimn_loss, compress_term, compress_loss, scalars)
    
    def otherterms(self, Z, label, label_s, V_old, V_neig, Y):
        Pi = F.one_hot(label, self.num_class).to(Z.device)
        Z = Z.T
        d, n = Z.shape
        term1 = 0.
        term2 = 0.
        for k in label_s:
            nei = self.total_receinodes_perclass[k]
            neig_idx = [self.neig.index(ii) for ii in nei]
            Z_ = Z[:, Pi[:, k] == 1]
            term1 += 0. if Pi[:, k].sum() == 0 else sum([(Y[k*d:(k+1)*d, :].T @ ( Z_ @ Z_.T/Z_.shape[1]))[i, i] for i in range(d)]) 
            term2 += 0. if Pi[:, k].sum() == 0 else self.rho/2.0*sum([((torch.norm(Z_ @ Z_.T/Z_.shape[1] - \
                                                  1.0/2.0*(V_old[k*d:(k+1)*d, :] + V_neig[j][k*d:(k+1)*d, :]), p='fro'))**2) for j in neig_idx])
        
        return term1, term2
