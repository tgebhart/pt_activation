import os
import parse
import pickle
import copy
import math
import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import dionysus as dion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import cross_val_score
import sklearn
import networkx as nx
import seaborn as sns

from pt_activation.models.fff import FFF as FFFRelu
from pt_activation.models.simple_mnist import CFF as CFFRelu
from pt_activation.models.simple_mnist_sigmoid import CFF as CFFSigmoid
from pt_activation.models.ccff import CCFF as CCFFRelu


COLORS = ['#12355b', '#ff6978']
EDGE_COLOR = '#272d2d'
PLT_LABELS = ['Unaltered', 'Adversarial']

def get_adv_info(filename):
    format_string = 'true-{}_adv-{}_sample-{}.npy'
    parsed = parse.parse(format_string, filename)
    return {'true class':int(parsed[0]), 'adv class':int(parsed[1]), 'sample':int(parsed[2])}

def read_adversaries(loc):
    ret = []
    for f in os.listdir(loc):
        if os.path.isfile(os.path.join(loc,f)) and f.find('.npy') != -1:
            adv = np.load(os.path.join(loc, f))
            info = get_adv_info(f)
            info['adversary'] = adv
            ret.append(info)
    return ret


def create_sample_graph(f):
    m = dion.homology_persistence(f)
    dgms = dion.init_diagrams(m,f)
    subgraphs = {}
    for i,c in enumerate(m):
        if len(c) == 2:
            if f[c[0].index][0] in subgraphs:
                subgraphs[f[c[0].index][0]].add_edge(f[c[0].index][0],f[c[1].index][0],weight=f[i].data)
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

    return subgraphs, dgms[0]


def create_lifetimes(dgms):
    return [[pt.birth - pt.death for pt in dgm if pt.death < np.inf] for dgm in dgms]


def create_subgraphs(model, batch_size, up_to, test_loader):
    device = torch.device("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    t = 0
    res_df = []
    subgraphs = []
    diagrams = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hiddens = model(data, hiddens=True)
            test_loss = F.nll_loss(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            for s in range(data.shape[0]):
                this_hiddens = [hiddens[i][s] for i in range(len(hiddens))]
                print('Filtration: {}'.format(s+t))
                f = model.compute_dynamic_filtration(data[s], this_hiddens)
                sg, dg = create_sample_graph(f)
                row = {'loss':test_loss, 'class':target.cpu().numpy()[s], 'prediction':pred.cpu().numpy()[s][0]}
                res_df.append(row)
                subgraphs.append(sg)
                diagrams.append(dg)

            t += batch_size
            if t >= up_to:
                break

    return pd.DataFrame(res_df), subgraphs, diagrams


def create_adversary_subgraphs(model, batch_size, up_to, adversaries):
    device = torch.device("cpu")
    adv_images = torch.tensor(np.array([a['adversary'] for a in adversaries]))
    adv_labels = torch.tensor(np.array([a['true class'] for a in adversaries]))
    adv_samples = [a['sample'] for a in adversaries]

    print(adv_images.shape, adv_labels.shape)

    advs = torch.utils.data.TensorDataset(adv_images, adv_labels)
    test_loader = torch.utils.data.DataLoader(advs, batch_size=batch_size, shuffle=False)

    model.eval()
    test_loss = 0
    correct = 0
    t = 0
    res_df = []
    subgraphs = []
    diagrams = []
    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hiddens = model(data, hiddens=True)
            test_loss = F.nll_loss(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for s in range(data.shape[0]):
                this_hiddens = [hiddens[i][s] for i in range(len(hiddens))]
                print('Filtration: {}'.format(s+t))
                f = model.compute_dynamic_filtration(data[s], this_hiddens)
                sg, dg = create_sample_graph(f)
                row = {'loss':test_loss, 'class':target.cpu().numpy()[s], 'prediction':pred.cpu().numpy()[s][0], 'sample':adv_samples[t+s]}
                res_df.append(row)
                subgraphs.append(sg)
                diagrams.append(dg)

            t += (batch_size)
            if t >= up_to:
                break

    return pd.DataFrame(res_df), subgraphs, diagrams



def run(adv_directory_loc, model, figure_loc, test_loader, some_num=10, of_int=1, num_filtration=2000):

    adversaries = read_adversaries(adv_directory_loc)
    adversaries = sorted(adversaries,  key=lambda k: k['sample'])

    res_df, sample_graphs, dgms = create_subgraphs(model, 1, num_filtration, test_loader)
    adv_df, adv_sample_graphs, adv_dgms = create_adversary_subgraphs(model, 1, num_filtration, adversaries)

    with open(os.path.join(adv_directory_loc, 'sample_graphs.pkl'), 'wb') as f:
        pickle.dump(sample_graphs, f)
    with open(os.path.join(adv_directory_loc, 'adv_sample_graphs.pkl'), 'wb') as f:
        pickle.dump(adv_sample_graphs, f)

    incorrects = res_df[res_df['class'] != res_df['prediction']]
    corrects = res_df[res_df['class'] == res_df['prediction']]

    correct_list = list(corrects.index)
    adv_correct_list = list(adv_df[~np.isin(adv_df['sample'], list(incorrects.index))].index)

    sample_graphs = [sample_graphs[i] for i in correct_list]
    adv_sample_graphs = [adv_sample_graphs[i] for i in adv_correct_list]

    sgl = np.zeros(len(sample_graphs))
    for i in range(len(sample_graphs)):
        sgl[i] = len(sample_graphs[i])
    adv_sgl = np.zeros(len(adv_sample_graphs))
    for i in range(len(adv_sample_graphs)):
        adv_sgl[i] = len(adv_sample_graphs[i])


    d = [[pt.birth, pt.death] for pt in adv_dgms[of_int] if pt.death < np.inf]
    d = np.array(d)
    ax = plt.subplot()
    ax.scatter(d[:,0], d[:,1], s=25, c=COLORS[0])
    lims = [
        np.min(.9*d[:,0]),  # min of both axes
        np.max(1.1*d[:,1]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('Birth Time')
    plt.ylabel('Death Time')
    plt.savefig(os.path.join(figure_loc, 'diagram.png'), dpi=1200)
    plt.close()

    d = [[pt.birth, pt.death] for pt in dgms[of_int] if pt.death < np.inf]
    d = np.array(d)
    ax = plt.subplot()
    ax.scatter(d[:,0], d[:,1], s=25, c=COLORS[1])
    lims = [
        np.min(d[:,0]),  # min of both axes
        np.max(d[:,1]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('Birth Time')
    plt.ylabel('Death Time')
    plt.savefig(os.path.join(figure_loc, 'adversary_diagram.png'), dpi=1200)
    plt.close()

    rng = [np.amin([np.amin(sgl), np.amin(adv_sgl)]), np.amax([np.amax(sgl), np.amax(adv_sgl)])]
    plt.hist([sgl, adv_sgl], bins='auto', color=COLORS, label=PLT_LABELS, range=rng)
    plt.ylabel('Number of Inputs')
    plt.xlabel('Number of Generators')
    plt.savefig(os.path.join(figure_loc, 'num_generators.png'), dpi=1200)
    plt.close()

    lifetimes = create_lifetimes(dgms)
    adv_lifetimes = create_lifetimes(adv_dgms)

    mls = [np.mean(l) for l in lifetimes]
    adv_mls = [np.mean(l) for l in adv_lifetimes]
    rng = [np.amin([np.amin(mls), np.amin(adv_mls)]), np.amax([np.amax(mls), np.amax(adv_mls)])]
    plt.hist([mls, adv_mls], bins='auto', range=rng, color=COLORS, label=PLT_LABELS)
    plt.ylabel('Number of Inputs')
    plt.xlabel('Mean Lifetime of Generators')
    plt.savefig(os.path.join(figure_loc, 'mean_lifetime.png'), dpi=1200)
    plt.close()

    all_gois = []
    for i in range(len(sample_graphs)):
        print('all_gois', i)
        a = [sample_graphs[i][k] for k in sample_graphs[i].keys()]
        all_gois.append(nx.compose_all(a))

    adv_all_gois = []
    for i in range(len(adv_sample_graphs)):
        print('adv_all_gois', i)
        a = [adv_sample_graphs[i][k] for k in adv_sample_graphs[i].keys()]
        adv_all_gois.append(nx.compose_all(a))

    eigs = []
    for i in range(len(all_gois)):
        print('normal eig', i)
        eigs.append(nx.linalg.laplacian_spectrum(all_gois[i]))

    adv_eigs = []
    for i in range(len(adv_all_gois)):
        print('adv eig', i)
        adv_eigs.append(nx.linalg.laplacian_spectrum(adv_all_gois[i]))

    mean_eigs = np.zeros((len(eigs),3))
    for i in range(len(eigs)):
        mean_eigs[i] = [eigs[i].mean(), eigs[i].shape[0], np.median(eigs[i])]

    adv_mean_eigs = np.zeros((len(adv_eigs),3))
    for i in range(len(adv_eigs)):
        adv_mean_eigs[i] = [adv_eigs[i].mean(), adv_eigs[i].shape[0], np.median(adv_eigs[i])]

    rng = [np.amin([np.amin(mean_eigs[:,0]), np.amin(adv_mean_eigs[:,0])]), np.amax([np.amax(mean_eigs[:,0]), np.amax(adv_mean_eigs[:,0])])]
    plt.hist([mean_eigs[:,0], adv_mean_eigs[:,0]], range=rng, bins='auto', color=COLORS, label=PLT_LABELS)
    plt.xlabel('Mean Eigenvalue')
    plt.ylabel('Number of Inputs')
    plt.savefig(os.path.join(figure_loc, 'mean_eigenvalues.png'), dpi=1200)
    plt.close()

    rng = [np.amin([np.amin(mean_eigs[:,1]), np.amin(adv_mean_eigs[:,1])]), np.amax([np.amax(mean_eigs[:,1]), np.amax(adv_mean_eigs[:,1])])]
    plt.hist([mean_eigs[:,1], adv_mean_eigs[:,1]], range=rng, bins='auto', color=COLORS, label=PLT_LABELS)
    plt.xlabel('Spectrum Size')
    plt.ylabel('Number of Inputs')
    plt.savefig(os.path.join(figure_loc, 'spectrum_size.png'), dpi=1200)
    plt.close()

    some_gois = []
    for i in range(len(sample_graphs)):
        print('some gois', i)
        sgik = list(sample_graphs[i].keys())
        a = [sample_graphs[i][k] for k in sgik[len(sgik)-some_num:]]
        some_gois.append(nx.compose_all(a))

    adv_some_gois = []
    for i in range(len(adv_sample_graphs)):
        print('adv some gois', i)
        sgik = list(adv_sample_graphs[i].keys())
        a = [adv_sample_graphs[i][k] for k in sgik[len(sgik)-some_num:]]
        adv_some_gois.append(nx.compose_all(a))

    some_eigs = []
    for i in range(len(some_gois)):
        print('some normal eigs ', i)
        some_eigs.append(nx.linalg.laplacian_spectrum(some_gois[i]))

    adv_some_eigs = []
    for i in range(len(adv_some_gois)):
        print('some adv eigs ', i)
        adv_some_eigs.append(nx.linalg.laplacian_spectrum(adv_some_gois[i]))

    plt.imshow(adversaries[of_int]['adversary'].reshape(28,28))
    plt.savefig(os.path.join(figure_loc, 'adversary.png'), dpi=1200)
    plt.close()

    options = {
    'node_color': COLORS[0],
    'node_size': 2,
    'width': 2,
    'with_labels':False}
    nx.draw_spring(some_gois[of_int], **options)
    plt.savefig(os.path.join(figure_loc, 'graph_bottom_10_generators.png'))
    plt.close()

    options = {
    'node_color': COLORS[1],
    'node_size': 2,
    'width': 2,
    'with_labels':False}
    nx.draw_spring(adv_some_gois[of_int], **options)
    plt.savefig(os.path.join(figure_loc, 'adv_graph_bottom_10_generators.png'))
    plt.close()

    some_gois2 = []
    for i in range(len(sample_graphs)):
        print(i)
        sgik = list(sample_graphs[i].keys())
        a = [sample_graphs[i][k] for k in sgik[:some_num]]
        some_gois2.append(nx.compose_all(a))

    adv_some_gois2 = []
    for i in range(len(adv_sample_graphs)):
        print(i)
        sgik = list(adv_sample_graphs[i].keys())
        a = [adv_sample_graphs[i][k] for k in sgik[:some_num]]
        adv_some_gois2.append(nx.compose_all(a))

    some_eigs2 = []
    for i in range(len(some_gois)):
        print('normal ', i)
        some_eigs2.append(nx.linalg.laplacian_spectrum(some_gois2[i]))

    adv_some_eigs2 = []
    for i in range(len(adv_some_gois)):
        print('adv ', i)
        adv_some_eigs2.append(nx.linalg.laplacian_spectrum(adv_some_gois2[i]))

    options = {
    'node_color': COLORS[0],
    'node_size': 2,
    'width': 2,
    'with_labels':False}
    nx.draw_spring(some_gois2[of_int], **options)
    plt.savefig(os.path.join(figure_loc, 'graph_top_10_generators.png'))
    plt.close()

    options = {
    'node_color': COLORS[1],
    'node_size': 2,
    'width': 2,
    'with_labels':False}
    nx.draw_spring(adv_some_gois2[of_int], **options)
    plt.savefig(os.path.join(figure_loc, 'adv_graph_top_10_generators.png'))
    plt.close()

    density = np.zeros(len(all_gois))
    adv_density = np.zeros(len(adv_all_gois))
    for i in range(len(all_gois)):
        density[i] = nx.classes.function.density(all_gois[i])
    for i in range(len(adv_all_gois)):
        adv_density[i] = nx.classes.function.density(adv_all_gois[i])

    rng = [np.amin([np.amin(density), np.amin(adv_density)]), np.amax([np.amax(density), np.amax(adv_density)])]
    plt.hist([density, adv_density], bins='auto', color=COLORS, label=PLT_LABELS, range=rng)
    plt.xlabel('Density')
    plt.ylabel('Number of Inputs')
    plt.savefig(os.path.join(figure_loc, 'density.png'), dpi=1200)
    plt.close()

    print_out = {'average lifetime':np.mean(mls),
                'average adversary lifetime':np.mean(adv_mls),
                'average eigenvalue': np.mean(mean_eigs[:,0]),
                'average adversary eigenvalue': np.mean(adv_mean_eigs[:,0]),
                'average spectrum size': np.mean(mean_eigs[:,1]),
                'average adversary spectrum size': np.mean(adv_mean_eigs[:,1]),
                'average generator count': np.mean(sgl),
                'average adversary generator count': np.mean(adv_sgl),
                'average density': np.mean(density),
                'average adversary density': np.mean(adv_density),

                'std lifetime':np.std(mls),
                'std adversary lifetime':np.std(adv_mls),
                'std eigenvalue': np.std(mean_eigs[:,0]),
                'std adversary eigenvalue': np.std(adv_mean_eigs[:,0]),
                'std spectrum size': np.std(mean_eigs[:,1]),
                'std adversary spectrum size': np.std(adv_mean_eigs[:,1]),
                'std generator count': np.std(sgl),
                'std adversary generator count': np.std(adv_sgl),
                'std density': np.std(density),
                'std adversary density': np.std(adv_density)
                }
    pd.DataFrame([print_out]).to_csv(os.path.join(figure_loc, 'stats.csv'))


def main():


    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('-m', '--model-location', type=str, required=True,
                        help='location of stored trained model')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='which dataset to use (mnist or fashion)')
    parser.add_argument('-mt', '--model-type', type=str, required=True,
                        help='which model architecture to use (FFFRelu, CFFRelu, CFFSigmoid, CCFFRelu, etc)')
    parser.add_argument('-a', '--adversary-directory', type=str, required=True,
                        help='where to find adversaries in .npy format')
    parser.add_argument('-o', '--figure-loc', type=str, required=True,
                        help='where to save output results / graphs')
    parser.add_argument('-nf', '--number-filtration', type=int, required=False, default=2000,
                        help='number of filtrations to consider for each unaltered and adversarial examples')




    args = parser.parse_args()

    mt = args.model_type
    if mt == 'CFFRelu':
        model = CFFRelu()
    if mt == 'CFFSigmoid':
        model = CFFSigmoid()
    if mt == 'CCFFRelu':
        model = CCFFRelu()
    if mt == 'FFFRelu':
        model = FFFRelu()
    model.load_state_dict(torch.load(args.model_location))

    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.dataset == 'mnist':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=1, shuffle=False, **kwargs)
    if args.dataset == 'fashion':
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=1, shuffle=False, **kwargs)


    run(args.adversary_directory, model, args.figure_loc, test_loader, num_filtration=args.number_filtration)


if __name__ == '__main__':
    main()

# end
