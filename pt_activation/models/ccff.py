import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import dionysus as dion

import numpy as np
import pandas as pd

from pt_activation.functions.filtration import conv_filtration_fast2, linear_filtration_fast2, max_pooling_filtration, conv_layer_as_matrix, spec_hash, abs_sort

class CCFF(nn.Module):
    def __init__(self, num_classes=10):
        super(CCFF, self).__init__()

        self.c1 = nn.Conv2d(1, 3, kernel_size=5, stride=1, bias=False)
        self.c2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, bias=False)

        self.l1 = nn.Linear(1452, 256, bias=False)
        self.l2 = nn.Linear(256, num_classes, bias=False)

        self.params = [self.c1,self.c2,self.l1,self.l2]


    def forward(self, x, hiddens=False):

        h1 = torch.relu(self.c1(x))
        h2 = torch.relu(self.c2(h1))
        resized = h2.view(h2.size(0), -1)
        h3 = torch.relu(self.l1(resized))
        y = self.l2(h3)
        hiddens = [h1, resized, h3, y]
        if hiddens:
            return F.log_softmax(y, dim=1), hiddens
        return F.log_softmax(y, dim=1)


    def save_string(self):
        return "ccff.pt"

    # def layerwise_ids(self, input_size=28*28):
    #     l1_size = (28-self.kernel_size+1)**2*self.filters
    #     l1_end = input_size+l1_size
    #     l2_end = l1_end+self.fc1_size
    #     l3_end = l2_end + 10
    #     return [range(input_size), range(input_size, l1_end), range(l1_end, l2_end), range(l2_end, l3_end)]

    def compute_static_filtration(self, x, hiddens, percentile=None):
        x_id = 0

        f = dion.Filtration()
        mat = np.absolute(conv_layer_as_matrix(self.conv1.weight.data, x, self.conv1.stride[0]))
        x = x.cpu().detach().numpy().reshape(-1)

        if percentile is None:
            percentile_1 = 0
        else:
            percentile_1 = np.percentile(mat, percentile)
        gtzx = np.argwhere(x > 0)

        h1_id_start = x.shape[0]
        h1_births = np.zeros(mat.shape[0])
        # loop over each entry in the reshaped (column) x vector
        for xi in gtzx:
            # compute the product of each filter value with current x in iteration.
            all_xis = mat[:,xi]
            max_xi = all_xis.max()
            # set our x filtration as the highest product
            f.append(dion.Simplex([xi], max_xi))
            gtpall_xis = np.argwhere(all_xis > percentile_1)[:,0]
            # iterate over all products
            for mj in gtpall_xis:
                # if there is another filter-xi combination that has a higher
                # product, save this as the birth time of that vertex.
                if h1_births[mj] < all_xis[mj]:
                    h1_births[mj] = all_xis[mj]
                f.append(dion.Simplex([xi, mj+h1_id_start], all_xis[mj]))

        h1 = hiddens[0].cpu().detach().numpy()
        h2_id_start = h1_id_start + h1.shape[0]
        mat = np.absolute(self.fc1.weight.data.cpu().detach().numpy())
        h2_births = np.zeros(mat.shape[0])

        if percentile is None:
            percentile_2 = 0
        else:
            percentile_2 = np.percentile(mat, percentile)
        gtzh1 = np.argwhere(h1 > 0)

        for xi in gtzh1:
            all_xis = mat[:,xi]
            max_xi = all_xis.max()
            if h1_births[xi] < max_xi:
                h1_births[xi] = max_xi
            gtpall_xis = np.argwhere(all_xis > percentile_2)[:,0]

            for mj in gtpall_xis:
                if h2_births[mj] < all_xis[mj]:
                    h2_births[mj] = all_xis[mj]
                f.append(dion.Simplex([xi+h1_id_start, mj+h2_id_start], all_xis[mj]))


        # now add maximum birth time for each h1 hidden vertex to the filtration.
        for i in np.argwhere(h1_births > 0):
            f.append(dion.Simplex([i+h1_id_start], h1_births[i]))


        h2 = hiddens[1].cpu().detach().numpy()
        h3_id_start = h2_id_start + h2.shape[0]
        mat = np.absolute(self.fc2.weight.data.cpu().detach().numpy())
        h3_births = np.zeros(mat.shape[0])

        if percentile is None:
            percentile_3 = 0
        else:
            percentile_3 = np.percentile(mat, percentile)
        gtzh2 = np.argwhere(h2 > 0)

        for xi in gtzh2:
            all_xis = mat[:,xi]
            max_xi = all_xis.max()
            if h2_births[xi] < max_xi:
                h2_births[xi] = max_xi
            gtpall_xis = np.argwhere(all_xis > percentile_3)[:,0]

            for mj in gtpall_xis:
                if h3_births[mj] < all_xis[mj]:
                    h3_births[mj] = all_xis[mj]
                f.append(dion.Simplex([xi+h2_id_start, mj+h3_id_start], all_xis[mj]))


        # now add maximum birth time for each h2 hidden vertex to the filtration.
        for i in np.argwhere(h2_births > 0):
            f.append(dion.Simplex([i+h2_id_start], h2_births[i]))

        # now add maximum birth time for each h3 hidden vertex to the filtration.
        for i in np.argwhere(h3_births > 0):
            f.append(dion.Simplex([i+h3_id_start], h3_births[i]))

        print('filtration size', len(f))
        print('Sorting filtration...')
        f.sort(reverse=True)
        return f

    def compute_dynamic_filtration_batch(self, x, hiddens, percentile=None):
        '''Generally too memory intensive to store entire batch of filtrations.
        Instead iterate over each example input, compute diagram, then save.
        '''
        filtrations = []
        for s in range(x.shape[0]):
            # check if this makes sense
            this_hiddens = [hiddens[0][s], hiddens[1][s], hiddens[2][s]]
            print('Filtration: {}'.format(s))
            print(hiddens[0].shape, hiddens[1].shape, hiddens[2].shape)
            f = self.compute_dynamic_filtration(x[s,0], this_hiddens, percentile=percentile)
            filtrations.append(f)
        return filtrations

    def compute_dynamic_filtration(self, x, hiddens, percentile=None):
        f = dion.Filtration()
        id_start = 0
        num_channels = x.shape[0]

        for c in range(num_channels):

            s = x[c].cpu().detach().numpy().reshape(-1).shape[0]
            h1_id_start = id_start + s
            f, h1_births = conv_filtration(f, x[c], self.c1.weight.data[:,c,:,:], id_start, h1_id_start, percentile=percentile, stride=self.c1.stride[0])


            h1_births = h1_births.reshape(hiddens[0].shape)
            h2_id_start = h1_id_start + hiddens[0].cpu().detach().numpy().flatten().shape[0]
            start_2 = h2_id_start
            h2_births = []
            for d in range(hiddens[0].shape[0]):
                start_1 = h1_id_start + (h1_births[d].flatten().shape[0]*d)
                f, h2b = conv_filtration(f, hiddens[0][d], self.c2.weight.data[:,d,:,:], start_1, start_2, h0_births=h1_births[d], percentile=percentile, stride=self.c1.stride[0])
                start_2 += h2b.shape[0]
                h2_births.append(h2b)
            h2_births = np.array(h2_births).reshape(-1)

            h3_id_start = h2_id_start + hiddens[1].cpu().detach().numpy().shape[0]
            f, h3_births = linear_filtration(f, hiddens[1], self.l1, h2_births, h2_id_start, h3_id_start, percentile=percentile, last=False)

            h4_id_start = h3_id_start + hiddens[2].cpu().detach().numpy().shape[0]
            f = linear_filtration(f, hiddens[2], self.l2, h3_births, h3_id_start, h4_id_start, percentile=percentile, last=True)

        print('filtration size', len(f))
        print('Sorting filtration...')
        f.sort(reverse=True)
        return f

    def compute_dynamic_filtration2(self, x, hiddens, percentile=0, return_nm=False, absolute_value=True, input_layer=False):
        id = 0
        f = dion.Filtration()
        nm = {}
        wm = {}
        params = self.params
        percentiles = np.zeros((len(params)))
        for l in range(len(params)):
            percentiles[l] = (1/(l+1))*np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)

        def collect_result(res):
            nonlocal id
            nonlocal f
            nonlocal nm
            nonlocal wm
            for enum in res:
                nodes = enum[0]
                weight = enum[1][0]
                if len(nodes) == 1:
                    if nodes[0] not in nm:
                        nm[nodes[0]] = id
                        id += 1
                        f.append(dion.Simplex([nm[nodes[0]]], weight))
                    else:
                        f.append(dion.Simplex([nm[nodes[0]]], weight))
                if len(nodes) == 2:
                    act_weight = enum[1][1]
                    if nodes[0] not in nm:
                        nm[nodes[0]] = id
                        id += 1
                    if nodes[1] not in nm:
                        nm[nodes[1]] = id
                        id += 1
                    wm[(nodes[0],nodes[1])] = act_weight
                    f.append(dion.Simplex([nm[nodes[0]], nm[nodes[1]]], weight))



        x = x.cpu().detach().numpy()
        num_channels = x.shape[0]
        l = 0
        hn = hiddens[l].cpu().detach().numpy()
        nlc = hn.shape[0]
        nls = hn.shape[1]*hn.shape[2]
        stride = 1
        enums = []
        for c in range(num_channels):
            p = params[l].weight.data[:,c,:,:]
            mat = conv_layer_as_matrix(p, x[c], stride)
            m1, h0_births, h1_births = conv_filtration_fast2(x[c], mat, l, c, nlc, nls, percentile=percentiles[l], absolute_value=absolute_value)
            if input_layer:
                enums += m1
                enums += [([spec_hash((l,c,i[0]))], [h0_births[i].item()]) for i in np.argwhere(h0_births > percentiles[l])]
            enums += [([spec_hash((l+1,i[0]//nlc,i[0]%nls))], [h1_births[i].item()]) for i in np.argwhere(h1_births > percentiles[l])]
        collect_result(enums)

        h1 = hiddens[l].cpu().detach().numpy()
        num_channels = h1.shape[0]
        l = 1
        hn = hiddens[l].cpu().detach().numpy()
        nlc = hn.reshape((hn.shape[0],-1)).shape[1]
        nls = hn.shape[0]
        stride = 1
        enums = []
        for c in range(num_channels):
            p = params[l].weight.data[:,c,:,:]
            mat = conv_layer_as_matrix(p, h1[c], stride)
            m1, h0_births, h1_births = conv_filtration_fast2(h1[c], mat, l, c, nlc, nls, percentile=percentiles[l], absolute_value=absolute_value)
            enums += m1
            comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
            enums += [([spec_hash((l,c,i[0]))], [h0_births[i].item()]) for i in np.argwhere(h0_births > comp_percentile)]
            enums += [([spec_hash((l+1,0,i[0]))], [h1_births[i].item()]) for i in np.argwhere(h1_births > percentiles[l])]
        collect_result(enums)

        h1 = hiddens[l].cpu().detach().numpy()
        l = 2
        p = params[l]
        percentiles[l] = np.percentile(np.absolute(h1*p.weight.data.cpu().detach().numpy()), percentile)
        m1, h0_births, h1_births = linear_filtration_fast2(h1, p, l, 0, percentile=percentiles[l], absolute_value=absolute_value)
        enums = m1
        comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
        enums += [([spec_hash((l,0,i[0]))], [h0_births[i]]) for i in np.argwhere(h0_births > comp_percentile)]

        h1 = hiddens[l].cpu().detach().numpy()
        l = 3
        p = params[l]
        percentiles[l] = np.percentile(np.absolute(h1*p.weight.data.cpu().detach().numpy()), percentile)
        m1, h0_births, h1_births_2 = linear_filtration_fast2(h1, p, l, 0, percentile=percentiles[l], absolute_value=absolute_value)
        enums += m1

        max1 = np.maximum.reduce([h0_births, h1_births])
        comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
        enums += [([spec_hash((l,0,i[0]))], [max1[i]]) for i in np.argwhere(max1 > comp_percentile)]
        enums += [([spec_hash((l+1,0,i[0]))], [h1_births_2[i]]) for i in np.argwhere(h1_births_2 > percentiles[l])]

        collect_result(enums)

        print('filtration size', len(f))

        f.sort(reverse=True)
        if return_nm:
            return f, nm, wm
        else:
            return f

def compute_percentiles(h, percentile, absolute_value):
    if absolute_value:
        return np.percentile(np.absolute(h), percentile)
    else:
        return np.percentile(h, percentile)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,_ = model(data, hiddens=False)
        closs = nn.CrossEntropyLoss()
        loss = closs(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_homology(args, model, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, hiddens = model(data, hiddens=True)
        loss = F.nll_loss(output, target)
        loss.backward()
        reg_loss = homology_regularizer()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_ = model(data,hiddens=False)
            closs = nn.CrossEntropyLoss(reduction='sum')
            test_loss += closs(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_homology(args, model, device, test_loader, epoch, res_df):

    model.eval()
    test_loss = 0
    correct = 0
    # capture a hidden state to use later, we don't care about its values though
    hiddens = None
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hiddens = model(data, hiddens=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    this_hiddens = [hiddens[0][0], hiddens[1][0], hiddens[2][0]]
    f = model.compute_static_filtration(data[0,0], this_hiddens)
    m = dion.homology_persistence(f)
    dgms = dion.init_diagrams(m, f)
    row = {'diagrams':dgms, 'loss':test_loss, 'epoch':epoch, 'accuracy':100. * correct / len(test_loader.dataset)}
    res_df.append(row)
    return res_df

def create_diagrams(args, model):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.test_batch_size, shuffle=False, **kwargs)


    df_filename = model.save_string()
    df_filename = str(args.up_to) + '_' + df_filename[:df_filename.find('.pt')] + '.pkl'
    df_loc = os.path.join(args.diagram_directory, df_filename)

    model.eval()
    test_loss = 0
    correct = 0
    t = 0
    res_df = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hiddens = model(data, hiddens=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for s in range(data.shape[0]):
                # check if this makes sense
                this_hiddens = [hiddens[0][s], hiddens[1][s], hiddens[2][s]]
                print('Filtration: {}'.format(s+t))
                f = model.compute_dynamic_filtration(data[s,0], this_hiddens)
                #
                m = dion.homology_persistence(f)
                dgms = dion.init_diagrams(m, f)
                row = {'diagrams':dgms, 'loss':output.cpu().numpy()[s][0], 'class':target.cpu().numpy()[s], 'prediction':pred.cpu().numpy()[s][0]}
                res_df.append(row)

            t += args.test_batch_size
            if t >= args.up_to:
                break

    res_df = pd.DataFrame(res_df)
    res_df.to_pickle(df_loc)

def persistence_score(d):
    diag = np.array([[pt.birth,pt.death] for pt in d])
    diag = diag[~np.isinf(diag[:,1])]
    return np.average(diag[:,0] - diag[:,1])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('-m', '--model-directory', type=str, required=True,
                        help='location to store trained model')
    parser.add_argument('-d', '--diagram-directory', type=str, required=False,
                        help='location to store homology info')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--up-to', type=int, default=500, metavar='N',
                        help='How many testing exmaples for creating diagrams')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-ct', '--create-diagrams', action='store_true', default=False,
                        help='Whether to compute homology on dynamic graph after training')
    parser.add_argument('-ht', '--homology-train', action='store_true', default=False,
                        help='Whether to compute homology on static graph during training')
    parser.add_argument('-da', '--dataset', type=str, required=True,
                        help='which dataset to train on (mnist or fashionmnist)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])), batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])), batch_size=args.test_batch_size, shuffle=False, **kwargs)


    if args.dataset == 'fashion':

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/fashion', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])), batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = CCFF().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    res_df = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        if args.homology_train:
            res_df = test_homology(args,model,device,test_loader, epoch, res_df)
        else:
            test(args, model, device, test_loader)

    if args.homology_train and args.diagram_directory is not None:
        df_filename = model.save_string()
        df_filename = 'train_homology_' + df_filename[:df_filename.find('.pt')] + '.pkl'
        df_loc = os.path.join(args.diagram_directory, df_filename)
        res_df = pd.DataFrame(res_df)
        res_df.to_pickle(df_loc)

    save_path = os.path.join(args.model_directory, model.save_string())
    torch.save(model.state_dict(), save_path)

    if args.diagram_directory is not None and args.create_diagrams:
        create_diagrams(args, model)

if __name__ == '__main__':
    main()
