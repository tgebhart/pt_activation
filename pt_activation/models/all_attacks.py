import os
import argparse

import foolbox
from foolbox.criteria import TargetClassProbability
import torch
import numpy as np
from torchvision import datasets, transforms

from pt_activation.models.simple_mnist_sigmoid import CFF as CFFSigmoid
from pt_activation.models.simple_mnist import CFF as CFFRelu
from pt_activation.models.fff import FFF as FFFRelu
from pt_activation.models.ccff import CCFF as CCFFRelu


def get_attack(attack, fmodel):
    if attack == 'lbfgsm':
        return foolbox.attacks.LBFGSAttack(fmodel)
    if attack == 'carliniwagnerl2':
        return foolbox.attacks.CarliniWagnerL2Attack(fmodel)
    if attack == 'projected_gradient_descent':
        return foolbox.attacks.ProjectedGradientDescentAttack(fmodel)
    if attack == 'gaussian_noise':
        return foolbox.attacks.AdditiveGaussianNoiseAttack(fmodel)

def run(attack_name, adversary_location, model, dataset, classes=list(range(10)), upto=4000):

    if dataset == 'mnist':
        mnist_dataset = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
    if dataset == 'fashion':
        mnist_dataset = datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.ToTensor())

    if not os.path.exists(adversary_location):
        os.makedirs(adversary_location)

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    model.eval()
    fmodel = foolbox.models.PyTorchModel(model, num_classes=10, bounds=(0,255))

    # for c in classes:
    if upto > len(mnist_dataset):
        upto = len(mnist_dataset)

    for i in range(upto):
        image, label = mnist_dataset[i]

        label = label.data.numpy()

        print('sample: {}, label {}'.format(i, label))
        image = image.data.numpy()

        # apply attack on source image

        attack = get_attack(attack_name, fmodel)
        adversarial = attack(image, label)
        if adversarial is not None:
            c = np.argmax(fmodel.predictions(adversarial))

            save_name = 'true-{}_adv-{}_sample-{}.npy'.format(label, c, i)
            save_loc = os.path.join(adversary_location, save_name)
            np.save(save_loc, adversarial)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('-m', '--model-directory', type=str, required=True,
                        help='location to store trained model')
    parser.add_argument('-mt', '--model-type', type=str, required=True,
                        help='which model architecture to use')
    parser.add_argument('-da', '--dataset', type=str, required=True,
                        help='which dataset to use (mnist or fashion)')
    parser.add_argument('-mn', '--model-name', type=str, required=True,
                        help='name of saved mode (should end in .pt)')
    parser.add_argument('-d', '--adversary-directory', type=str, required=True,
                        help='location to store adversaries')
    parser.add_argument('-ut', '--up-to', type=int, required=False, default=4000,
                        help='number of adversaries to create')
    parser.add_argument('-c', '--classes', required=False, default=list(range(10)), nargs='+',
                        help='which classes to create adversaries for')


    args = parser.parse_args()
    classes = list(map(lambda x : int(x), args.classes))

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
    model.load_state_dict(torch.load(model_location))

    attacks = ['lbfgsm', 'carliniwagnerl2', 'projected_gradient_descent', 'gaussian_noise']

    for attack in attacks:

        adversary_location = os.path.join(args.adversary_directory, attack, args.model_name)

        run(attack, adversary_location, model, args.dataset, classes=classes, upto=args.up_to)


if __name__ == '__main__':
    main()
