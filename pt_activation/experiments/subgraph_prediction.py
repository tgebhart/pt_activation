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
import scipy
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import manifold
from collections import OrderedDict
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
                row = {'loss':test_loss, 'class':target.cpu().numpy()[s], 'prediction':pred.cpu().numpy()[s][0]}
                res_df.append(row)
                subgraphs.append(sg)
                diagrams.append(dg)

            t += (batch_size)
            if t >= up_to:
                break

    return pd.DataFrame(res_df), subgraphs, diagrams






def run(adv_directory_loc, model, figure_loc, test_loader, labels, some_num=10, of_int=1, num_filtration=3000, graphs_precomputed=True, take=-1):
    colors = ['black', 'blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'silver', 'cyan']
    adversaries = read_adversaries(adv_directory_loc)
    adversaries = sorted(adversaries,  key=lambda k: k['sample'])

    if graphs_precomputed:
        sample_graphs = pickle.load( open(os.path.join(adv_directory_loc, 'sample_graphs.pkl'), "rb") )
        adv_sample_graphs = pickle.load( open(os.path.join(adv_directory_loc, 'adv_sample_graphs.pkl'), "rb") )
    else:
        res_df, sample_graphs, dgms = create_subgraphs(model, 1, num_filtration, test_loader)
        adv_df, adv_sample_graphs, adv_dgms = create_adversary_subgraphs(model, 1, num_filtration, adversaries)

    kernel = 'linear'
    edges = set()
    for i in range(len(sample_graphs)):
        for k in list(sample_graphs[i].keys())[:take]:
            for x in sample_graphs[i][k].edges(data=True):
                edge_name = str(x[0])+'-'+str(x[1])
                edges.add(edge_name)

    edf = pd.DataFrame(np.zeros((len(sample_graphs),len(edges))), columns=list(edges))
    for i in range(len(sample_graphs)):
        print('Sample: {}/{}'.format(i,len(sample_graphs)))
        lst = list(sample_graphs[i].keys())
        for k in lst[:take]:
            for x in sample_graphs[i][k].edges(data=True):
                edge_name = str(x[0])+'-'+str(x[1])
                edf.iloc[i][edge_name] += 1

    X = edf.values
    y = res_df['class'].values

    print('Cross Val SVM ...')
    clf = svm.SVC( decision_function_shape='ovo', kernel=kernel)
    cv_scores = cross_val_score(clf, X, y, cv=10)

    natural_performance = res_df[res_df['class'] == res_df['prediction']].shape[0]/res_df.shape[0]

    print('Fitting SVM ...')
    t_fit = svm.SVC(decision_function_shape='ovo', kernel=kernel).fit(X,y)

    adv_edf = pd.DataFrame(np.zeros((len(adv_sample_graphs),len(edges))), columns=list(edges))
    for i in range(len(adv_sample_graphs)):
        print('Sample: {}/{}'.format(i,len(adv_sample_graphs)))
        lst = list(adv_sample_graphs[i].keys())
        for k in lst[:take]:
            for x in adv_sample_graphs[i][k].edges(data=True):
                edge_name = str(x[0])+'-'+str(x[1])
                if edge_name in adv_edf.columns:
                    adv_edf.iloc[i][edge_name] += 1

    print('Predicting SVM ...')
    adv_preds = t_fit.predict(adv_edf.values)

    recovery_accuracy = adv_df[adv_df['class'] == adv_preds].shape[0]/adv_df.shape[0]
    adversary_retention = adv_df[adv_df['prediction'] == adv_preds].shape[0]/adv_df.shape[0]

    plt.imshow(adversaries[of_int]['adversary'].reshape(28,28))
    plt.savefig(os.path.join(figure_loc, 'adversary.png'), dpi=1200)
    plt.close()

    fig, ax = plt.subplots()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print('PCA...')
    for i in range(len(X_pca)):
        mark = "o" if res_df.iloc[i]['prediction'] == res_df.iloc[i]['class'] else "x"
        ax.scatter(X_pca[i,0], X_pca[i,1], color=colors[res_df['class'].iloc[i]], label=labels[res_df['class'].iloc[i]], marker=mark)
    handles, labs = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labs, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.savefig(os.path.join(figure_loc, 'pca.png'), dpi=1200)
    plt.close()

    print('ISOMAP...')
    fig, ax = plt.subplots()
    X_dimmed = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    # X_dimmed = manifold.TSNE(n_components=2, init='pca', random_state=5).fit_transform(X)
    # X_dimmed = manifold.SpectralEmbedding(n_neighbors=100, n_components=2).fit_transform(X)
    # X_dimmed = manifold.MDS(2, max_iter=200, n_init=10).fit_transform(X)
    # X_dimmed = manifold.LocallyLinearEmbedding(10, 2, eigen_solver='auto', method='standard').fit_transform(X)
    for i in range(len(X_dimmed)):
        mark = "o" if res_df.iloc[i]['prediction'] == res_df.iloc[i]['class'] else "x"
        ax.scatter(X_dimmed[i,0], X_dimmed[i,1], color=colors[res_df['class'].iloc[i]], label=labels[res_df['class'].iloc[i]], marker=mark)
    handles, labs = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labs, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.savefig(os.path.join(figure_loc, 'isomap.png'), dpi=1200)
    plt.close()

    classes = list(range(10))
    avgs = []
    for c in classes:
        avgs.append(edf.iloc[list(res_df[res_df['class'] == c].index)].mean(axis=0))
    average_df = pd.DataFrame(avgs)

    inps = np.zeros((average_df.shape[0], average_df.shape[0]))
    for i in range(inps.shape[0]):
        for j in range(inps.shape[0]):
            inps[i,j] = 1-scipy.spatial.distance.cosine(average_df.iloc[i], average_df.iloc[j])

    sns.heatmap(inps)
    plt.savefig(os.path.join(figure_loc, 'cosine_similarity.png'), dpi=1200)
    plt.close()

    print_out = {'average cv scores' : np.mean(cv_scores),
                'std cv scores': np.std(cv_scores),
                'natural performance': natural_performance,
                'recovery accuracy' : recovery_accuracy,
                'adversary retention' : adversary_retention,
                'take': take
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
    parser.add_argument('-nf', '--number-filtration', type=int, required=False, default=3000,
                        help='number of filtrations to consider for each unaltered and adversarial examples')
    parser.add_argument('-rg', '--recompute-graphs', action='store_true', default=False,
                    help='Whether to recreate subgraphs or load from adversary directory')
    parser.add_argument('-t', '--take', type=int, required=False, default=-1,
                        help='top number of subgraphs to consider')



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
        labels = list(range(10))
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=1, shuffle=False, **kwargs)
    if args.dataset == 'fashion':
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])), batch_size=1, shuffle=False, **kwargs)


    run(args.adversary_directory, model, args.figure_loc, test_loader, labels, num_filtration=args.number_filtration, take=args.take, graphs_precomputed=not args.recompute_graphs)


if __name__ == '__main__':
    main()

# end
