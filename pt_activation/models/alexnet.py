import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import dionysus as dion
import networkx as nx
import matplotlib.pyplot as plt

import multiprocessing as mp

import numpy as np
import pandas as pd

from pt_activation.functions.filtration import conv_filtration_fast2, linear_filtration_fast2, max_pooling_filtration, conv_layer_as_matrix, spec_hash

id = 0
nm = {}
wm = {}
f = dion.Filtration()

def first_layer(x,p,l,c,percentile,stride,nlc,nls):
    mat = conv_layer_as_matrix(p, x, stride)
    m1, h0_births, h1_births = conv_filtration_fast2(x, mat, l, c, nlc, percentile=percentile)
    # enums = m1
    # enums += [([spec_hash((l,c,i[0]))], [h0_births[i].item()]) for i in np.argwhere(h0_births > percentile)]
    enums = []
    enums += [([spec_hash((l+1,i[0]//nlc,i[0]%nls))], [h1_births[i].item()]) for i in np.argwhere(h1_births > percentile)]
    return enums

def mid_conv(h1,p,l,c,percentiles,stride,nlc,nls):
    mat = conv_layer_as_matrix(p, h1, stride)
    m1, h0_births, h1_births = conv_filtration_fast2(h1, mat, l, c, nlc, percentile=percentiles[l])
    enums = m1
    comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
    enums += [([spec_hash((l,c,i[0]))], [h0_births[i].item()]) for i in np.argwhere(h0_births > comp_percentile)]
    enums += [([spec_hash((l+1,i[0]//nlc,i[0]%nls))], [h1_births[i].item()]) for i in np.argwhere(h1_births > percentiles[l])]
    return enums

def max_pool_layer(h1,p,l,c,percentiles,h2):
    m1, h0_births, h1_births = max_pooling_filtration(h1, p, l, c, percentile=percentiles[l])
    enums = m1
    comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
    enums += [([spec_hash((l,c,i[0]))], [h0_births[i].item()]) for i in np.argwhere(h0_births > comp_percentile)]
    enums += [([spec_hash((l+1,c,i[0]))], [h2[i].item()]) for i in np.argwhere(h2 > percentiles[l])]
    # enums += [([spec_hash((l+1,c,i[0]))], [h1_births[i].item()]) for i in np.argwhere(h1_births > percentiles[l])]
    return enums

def last_pool(h1,p,l,c,percentiles):
    m1, h0_births, h1_births = max_pooling_filtration(h1, p, l, c, percentile=percentiles[l], linear_next=True)
    enums = m1
    comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
    enums += [([spec_hash((l,c,i[0]))], [h0_births[i].item()]) for i in np.argwhere(h0_births > comp_percentile)]
    # enums += [([spec_hash((l+1,0,i[0]+(c*h1_births.shape[0])))], [h1_births[i].item()]) for i in np.argwhere(h1_births > percentiles[l])]
    return enums

def collect_result(res):
    global nm
    global id
    global wm
    global f
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

def compute_induced_filtration_parallel(x, hiddens, params, percentile=0, stride=1, return_nm=True, absolute_value=True):

    pool = mp.Pool(mp.cpu_count())
    print('cpu count: {}'.format(mp.cpu_count()))

    global nm
    global id
    global f
    global wm

    percentiles = np.zeros((len(params)))
    for l in range(len(params)):
        percentiles[l] = (1/(l+1))*np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)

    x = x.cpu().detach().numpy()
    num_channels = x.shape[0]
    l = 0
    print('layer: {}'.format(l))
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    hn = hiddens[l].cpu().detach().numpy()
    nlc = hn.reshape((hn.shape[0],-1)).shape[1]
    nls = hn.shape[1]*hn.shape[2]
    for c in range(num_channels):
        p = params[l].weight.data[:,c,:,:]
        r = pool.apply_async(first_layer, args=(x[c],p,l,c,percentiles[l],stride,nlc,nls), callback=collect_result)
    pool.close()
    pool.join()

    h = hiddens[l].cpu().detach().numpy()
    num_channels = h.shape[0]
    l = 1
    percentiles[l] = np.percentile(np.absolute(h), percentile)
    hn = hiddens[l].cpu().detach().numpy()
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    print('layer: {}'.format(l))
    pool = mp.Pool(mp.cpu_count())
    for c in range(num_channels):
        h1 = h[c,:,:]
        hn2 = hn[c,:,:]
        p = params[l]
        r = pool.apply_async(max_pool_layer, args=(h1,p,l,c,percentiles,hn2), callback=collect_result)
    pool.close()
    pool.join()

    h = hiddens[l].cpu().detach().numpy()
    num_channels = h.shape[0]
    l = 2
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    hn = hiddens[l].cpu().detach().numpy()
    nlc = hn.reshape((hn.shape[0],-1)).shape[1]
    nls = hn.shape[1]*hn.shape[2]
    print('layer: {}'.format(l))
    pool = mp.Pool(mp.cpu_count())
    for c in range(num_channels):
        p = params[l].weight.data[:,c,:,:]
        h1 = h[c,:,:]
        r = pool.apply_async(mid_conv, args=(h1,p,l,c,percentiles,stride,nlc,nls), callback=collect_result)
    pool.close()
    pool.join()

    h = hiddens[l].cpu().detach().numpy()
    num_channels = h.shape[0]
    l = 3
    hn = hiddens[l].cpu().detach().numpy()
    percentiles[l] = np.percentile(np.absolute(h), percentile)
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    print('layer: {}'.format(l))
    pool = mp.Pool(mp.cpu_count())
    for c in range(num_channels):
        h1 = h[c,:,:]
        hn2 = hn[c,:,:]
        p = params[l]
        r = pool.apply_async(max_pool_layer, args=(h1,p,l,c,percentiles,hn2), callback=collect_result)
    pool.close()
    pool.join()

    h = hiddens[l].cpu().detach().numpy()
    num_channels = h.shape[0]
    l = 4
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    hn = hiddens[l].cpu().detach().numpy()
    nlc = hn.reshape((hn.shape[0],-1)).shape[1]
    nls = hn.shape[1]*hn.shape[2]
    print('layer: {}'.format(l))
    pool = mp.Pool(mp.cpu_count())
    for c in range(num_channels):
        p = params[l].weight.data[:,c,:,:]
        h1 = h[c,:,:]
        r = pool.apply_async(mid_conv, args=(h1,p,l,c,percentiles,stride,nlc,nls), callback=collect_result)
    pool.close()
    pool.join()

    h = hiddens[l].cpu().detach().numpy()
    num_channels = h.shape[0]
    l = 5
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    hn = hiddens[l].cpu().detach().numpy()
    nlc = hn.reshape((hn.shape[0],-1)).shape[1]
    nls = hn.shape[1]*hn.shape[2]
    print('layer: {}'.format(l))
    pool = mp.Pool(mp.cpu_count())
    for c in range(num_channels):
        p = params[l].weight.data[:,c,:,:]
        h1 = h[c,:,:]
        r = pool.apply_async(mid_conv, args=(h1,p,l,c,percentiles,stride,nlc,nls), callback=collect_result)
    pool.close()
    pool.join()

    h = hiddens[l].cpu().detach().numpy()
    num_channels = h.shape[0]
    l = 6
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    hn = hiddens[l].cpu().detach().numpy()
    nlc = hn.reshape((hn.shape[0],-1)).shape[1]
    nls = hn.shape[1]*hn.shape[2]
    print('layer: {}'.format(l))
    pool = mp.Pool(mp.cpu_count())
    for c in range(num_channels):
        p = params[l].weight.data[:,c,:,:]
        h1 = h[c,:,:]
        r = pool.apply_async(mid_conv, args=(h1,p,l,c,percentiles,stride,nlc,nls), callback=collect_result)
    pool.close()
    pool.join()

    h = hiddens[l].cpu().detach().numpy()
    num_channels = h.shape[0]
    l = 7
    percentiles[l] = np.percentile(np.absolute(h), percentile)
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    print('layer: {}'.format(l))
    pool = mp.Pool(mp.cpu_count())
    for c in range(num_channels):
        h1 = h[c,:,:]
        p = params[l]
        r = pool.apply_async(last_pool, args=(h1,p,l,c,percentiles), callback=collect_result)
    pool.close()
    pool.join()

    h1 = hiddens[l].cpu().detach().numpy()
    enums = [([spec_hash((l+1,0,i[0]))], [h1[i]]) for i in np.argwhere(h1 > percentiles[l])]
    l = 8
    print('layer: {}'.format(l))
    p = params[l]
    percentiles[l] = np.percentile(np.absolute(h1*p.weight.data.cpu().detach().numpy()), percentile)
    m1, h0_births, h1_births = linear_filtration_fast2(h1, p, l, 0, percentile=percentiles[l])
    enums += m1
    comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
    enums += [([spec_hash((l,0,i[0]))], [h0_births[i]]) for i in np.argwhere(h0_births > comp_percentile)]

    h1 = hiddens[l].cpu().detach().numpy()
    l = 9
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    print('layer: {}'.format(l))
    p = params[l]
    percentiles[l] = np.percentile(np.absolute(h1*p.weight.data.cpu().detach().numpy()), percentile)
    m1, h0_births, h1_births_9 = linear_filtration_fast2(h1, p, l, 0, percentile=percentiles[l])
    enums += m1

    max1 = np.maximum.reduce([h0_births, h1_births])
    comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
    enums += [([spec_hash((l,0,i[0]))], [max1[i]]) for i in np.argwhere(max1 > comp_percentile)]

    h1 = hiddens[l].cpu().detach().numpy()
    l = 10
    print('layer: {}'.format(l))
    # percentiles[l] = np.percentile(np.absolute(hiddens[l].cpu().detach().numpy()), percentile)
    p = params[l]
    percentiles[l] = np.percentile(np.absolute(h1*p.weight.data.cpu().detach().numpy()), percentile)
    m1, h0_births, h1_births_10 = linear_filtration_fast2(h1, p, l, 0, percentile=percentiles[l])
    enums += m1

    max1 = np.maximum.reduce([h0_births, h1_births_9])
    comp_percentile = percentiles[l-1] if percentiles[l-1] < percentiles[l] else percentiles[l]
    enums += [([spec_hash((l,0,i[0]))], [max1[i]]) for i in np.argwhere(max1 > comp_percentile)]
    enums += [([spec_hash((l+1,0,i[0]))], [h1_births_10[i]]) for i in np.argwhere(h1_births_10 > percentiles[l])]

    collect_result(enums)

    print('percentiles:', percentiles)

    # with open('quick_dump.txt', 'w') as fp:
    #     for k, v in nm.items():
    #         fp.write('{}, {}\n'.format(k,v))

    print('creating filtration object...')
    print('filtration size', len(f))
    print('Sorting filtration...')
    f.sort(reverse=True)
    if return_nm:
        return f, nm, wm
    else:
        return f



class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.stride = 1

        self.c1 = nn.Conv2d(3, 64, kernel_size=11, stride=1, bias=False)

        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.c2 = nn.Conv2d(64, 192, kernel_size=5, bias=False)

        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Conv2d(192, 384, kernel_size=3, bias=False)

        self.c4 = nn.Conv2d(384, 256, kernel_size=3, bias=False)

        self.c5 = nn.Conv2d(256, 256, kernel_size=3, bias=False)

        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l1 = nn.Linear(256, 4096, bias=False)

        self.l2 = nn.Linear(4096, 4096, bias=False)

        self.l3 = nn.Linear(4096, num_classes, bias=False)

        self.params = [self.c1,
                self.mp1,
                self.c2,
                self.mp2,
                self.c3,
                self.c4,
                self.c5,
                self.mp3,
                self.l1,
                self.l2,
                self.l3
                ]

    def forward(self, x, hiddens=False):

        h1 = torch.relu(self.c1(x))
        mp1 = self.mp1(h1)
        h2 = torch.relu(self.c2(mp1))
        mp2 = self.mp2(h2)

        h3 = torch.relu(self.c3(mp2))
        h4 = torch.relu(self.c4(h3))
        h5 = torch.relu(self.c5(h4))
        mp3 = self.mp3(h5)

        resized = mp3.view(mp3.size(0), -1)

        h6 = torch.relu(self.l1(resized))
        h7 = torch.relu(self.l2(h6))
        y = self.l3(h7)
        hiddens = [h1, mp1, h2, mp2, h3, h4, h5, resized, h6, h7, y]
        if hiddens:
            return y, hiddens
        return y

    def save_string(self, dataset):
        return "alexnet_{}.pt".format(dataset)

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

    def compute_induced_filtration(self, x, hiddens, percentile=0, return_nm=True, absolute_value=True):
        return compute_induced_filtration_parallel(x,hiddens,self.params,percentile=percentile,stride=self.stride, return_nm=return_nm, absolute_value=absolute_value)

    def compute_layer_mask(self, x, hiddens, subgraph_indices=[0], percentile=0):
        f,nm = self.compute_induced_filtration(x[0], hiddens, percentile=percentile)
        m = dion.homology_persistence(f)
        dgms = dion.init_diagrams(m,f)
        subgraphs = {}
        dion.plot.plot_diagram(dgms[0],show=False)
        plt.savefig('/home/tgebhart/projects/homo_explico/logdir/experiments/alexnet_vis/plt_3.png', format='png')
        plt.close()

        # compute ones tensors of same hidden dimensions
        muls = [torch.zeros(x.reshape((x.shape[0],-1)).shape)]

        for h in hiddens:
            print('h shape', h.reshape((h.shape[0],-1)).shape)
            muls.append(torch.zeros(h.reshape((h.shape[0],-1)).shape))

        fac = 1.0
        for i,c in enumerate(m):
            if len(c) == 2:
                if f[c[0].index][0] in subgraphs:
                    subgraphs[f[c[0].index][0]].add_edge(f[c[0].index][0],f[c[1].index][0], weight=f[i].data)
                else:
                    eaten = False
                    for k, v in subgraphs.items():
                        if v.has_node(f[c[0].index][0]):
                            v.add_edge(f[c[0].index][0], f[c[1].index][0], weight=f[i].data)
                            eaten = True
                            break
                    if not eaten:
                        g = nx.Graph()
                        g.add_edge(f[c[0].index][0], f[c[1].index][0], weight=f[i].data)
                        subgraphs[f[c[0].index][0]] = g

        #  I don't think we need this composition. Disjoint subgraphs are fine.
        # subgraph = nx.compose_all([subgraphs[k] for k in list(subgraphs.keys())[:thru]])

        ids = dict((v,k) for k,v in nm.items())

        ks = list(subgraphs.keys())
        print('KS: ',ks)
        ks_sub = [ks[s] for s in subgraph_indices]
        # lifetimes = np.empty(thru)
        # t = 0
        # for pt in dgms[0]:
        #     if pt.death < float('inf'):
        #         lifetimes[t] = pt.birth - pt.death
        #         t += 1
        #     if t >= thru:
        #         break
        # max_lifetime = max(lifetimes)
        # min_lifetime = min(lifetimes)

        for k in ks_sub:
            for e in subgraphs[k].edges(data=True):
                n1 = ids[e[0]]
                n2 = ids[e[1]]
                # allow the outcoming activation from that node at that channel (idx 1)
                if n1[1] > 0:
                    muls[n1[0]][n1[1],n1[2]] = fac
                else:
                    muls[n1[0]][n1[2]] = fac
                if n2[1] > 0:
                    muls[n2[0]][n2[1],n2[2]] = fac
                else:
                    muls[n2[0]][n2[2]] = fac

        return muls


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

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_ = model(data, hiddens=False)
            closs = nn.CrossEntropyLoss(reduction='sum')
            test_loss += closs(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch AlexNet on imagenet')
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
                        help='which dataset to train on (cifar)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == 'cifar':
        trainset = datasets.CIFAR10(root='../data/cifar', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='../data/cifar', train=False,
                                               download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=2)


    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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

    save_path = os.path.join(args.model_directory, model.save_string(args.dataset))
    torch.save(model.state_dict(), save_path)

    if args.diagram_directory is not None and args.create_diagrams:
        create_diagrams(args, model)

if __name__ == '__main__':
    main()
