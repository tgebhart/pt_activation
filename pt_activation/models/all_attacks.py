import os
import argparse

import foolbox
from foolbox.criteria import TargetClassProbability
import torch
import numpy as np
from torchvision import datasets, transforms

from pt_activation.models.cff_sigmoid import CFF as CFFSigmoid
from pt_activation.models.cff import CFF as CFFRelu
from pt_activation.models.fff import FFF as FFFRelu
from pt_activation.models.ccff import CCFF as CCFFRelu
from pt_activation.models.linear import FFF as FFF

from pt_activation.models.alexnet import AlexNet


def get_attack(attack, fmodel):
    if attack == 'lbfgsm':
        return foolbox.attacks.LBFGSAttack(fmodel)
    if attack == 'carliniwagnerl2':
        return foolbox.attacks.CarliniWagnerL2Attack(fmodel)
    if attack == 'carliniwagnerlinf':
        return foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=foolbox.distances.Linfinity)
    if attack == 'projected_gradient_descent':
        return foolbox.attacks.ProjectedGradientDescentAttack(fmodel)
    if attack == 'gaussian_noise':
        return foolbox.attacks.AdditiveGaussianNoiseAttack(fmodel)

def run(attack_name, adversary_location, model, dataset, classes=list(range(10)), upto=4000, max_iterations=1000):

    if dataset == 'mnist':
        mnist_dataset = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    if dataset == 'fashion':
        mnist_dataset = datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
    if dataset == 'cifar':
        mnist_dataset = datasets.CIFAR10(root='../data/cifar', train=True,
                                                download=True, transform = transforms.Compose(
                                                [transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    if not os.path.exists(adversary_location):
        os.makedirs(adversary_location)

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    model.eval()
    if dataset == 'mnist':
        fmodel = foolbox.models.PyTorchModel(model, num_classes=10, bounds=(0,255))
    else:
        fmodel = foolbox.models.PyTorchModel(model, num_classes=10, bounds=(-1,1))

    # for c in classes:
    if upto > len(mnist_dataset):
        upto = len(mnist_dataset)

    for i in range(upto):
        image, label = mnist_dataset[i]

        label = label

        print('sample: {}, label {}'.format(i, label))
        image = image.data.numpy()

        # apply attack on source image

        attack = get_attack(attack_name, fmodel)
        adversarial = attack(image, label, max_iterations=max_iterations)
        if adversarial is not None:
            c = np.argmax(fmodel.predictions(adversarial))

            save_name = 'true-{}_adv-{}_sample-{}.npy'.format(label, c, i)
            save_loc = os.path.join(adversary_location, save_name)
            np.save(save_loc, adversarial)



def main():
    # Training settings
    # attacks = ['lbfgsm', 'carliniwagnerl2', 'projected_gradient_descent', 'gaussian_noise', 'carliniwagnerlinf']
    attacks = ['lbfgsm', 'carliniwagnerl2', 'projected_gradient_descent', 'gaussian_noise']
    # attacks = ['carliniwagnerl2', 'projected_gradient_descent']
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('-m', '--model-directory', type=str, required=True,
                        help='location of stored trained model')
    parser.add_argument('-mt', '--model-type', type=str, required=True,
                        help='which model architecture to use')
    parser.add_argument('-da', '--dataset', type=str, required=True,
                        help='which dataset to use (mnist or fashion or cifar)')
    parser.add_argument('-mn', '--model-name', type=str, required=True,
                        help='name of saved mode (should end in .pt)')
    parser.add_argument('-d', '--adversary-directory', type=str, required=True,
                        help='location to store adversaries')
    parser.add_argument('-ut', '--up-to', type=int, required=False, default=4000,
                        help='number of adversaries to create')
    parser.add_argument('-c', '--classes', required=False, default=list(range(10)), nargs='+',
                        help='which classes to create adversaries for')
    parser.add_argument('-a','--attacks', nargs='+', help='List of attacks', required=False,
                        default=attacks)
    parser.add_argument('-mi', '--max-iterations', type=int, required=False, default=1000,
                        help='maximum number of iterations of attack')

    args = parser.parse_args()
    classes = list(map(lambda x : int(x), args.classes))
    attacks = args.attacks

    model_location = os.path.join(args.model_directory, args.model_name)
    mt = args.model_type
    if mt == 'CFFRelu':
        model = CFFRelu()
    if mt == 'CFFSigmoid':
        model = CFFSigmoid()
    if mt == 'CCFFRelu':
        model = CCFFRelu()
    if mt == 'FFFRelu':
        model = FFFRelu()
    if mt == 'FFF':
        model = FFF()
    if mt == 'AlexNet':
        model = AlexNet()
    model.load_state_dict(torch.load(model_location))

    for attack in attacks:

        adversary_location = os.path.join(args.adversary_directory, attack, args.model_name)

        run(attack, adversary_location, model, args.dataset, classes=classes, upto=args.up_to, max_iterations=args.max_iterations)


if __name__ == '__main__':
    main()
