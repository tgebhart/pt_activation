import os
import argparse

import foolbox
from foolbox.criteria import TargetClassProbability
import torch
import numpy as np
from torchvision import datasets, transforms

from pt_activation.models.ccff_fashion import CCFF

P = 0.9
ATTACK_NAME = 'lbfgsm'

def run(model_directory, model_name, adversary_directory, classes=list(range(10)), p=P, upto=5000):

    model_location = os.path.join(model_directory, model_name)
    model = CCFF()
    model.load_state_dict(torch.load(model_location))

    p_string = str(int(p*100))
    adversary_location = os.path.join(adversary_directory, ATTACK_NAME, model_name, p_string)
    if not os.path.exists(adversary_location):
        os.makedirs(adversary_location)

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}
    mnist_dataset = datasets.FashionMNIST('../data/fashion', train=False, download=True, transform=transforms.ToTensor())

    model.eval()
    fmodel = foolbox.models.PyTorchModel(model, num_classes=10, bounds=(0,255))

    # for c in classes:
    if upto > len(mnist_dataset):
        upto = len(mnist_dataset)
    for i in range(upto):
        image, label = mnist_dataset[i]

        label = label.data.numpy()

        print('adversary: {}, label {}'.format(i,label))
        image = image.data.numpy()

        # apply attack on source image
        attack = foolbox.attacks.LBFGSAttack(fmodel)
        adversarial = attack(image, label)
        if adversarial is not None:
            c = np.argmax(fmodel.predictions(adversarial))

            save_name = 'true-{}_adv-{}_sample-{}.npy'.format(label, c, i)
            save_loc = os.path.join(adversary_location, save_name)
            np.save(save_loc, adversarial)
        else:
            print('Could not create adversary')



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST')
    parser.add_argument('-m', '--model-directory', type=str, required=True,
                        help='location to store trained model')
    parser.add_argument('-mn', '--model-name', type=str, required=True,
                        help='name of saved mode (should end in .pt)')
    parser.add_argument('-d', '--adversary-directory', type=str, required=True,
                        help='location to store adversaries')
    parser.add_argument('-p', '--probability', type=float, required=False, default=P,
                        help='probability of adversarial examples')
    parser.add_argument('-c', '--classes', required=False, default=list(range(10)), nargs='+',
                        help='which classes to create adversaries for')
    parser.add_argument('-ut', '--up-to', type=int, required=False, default=P,
                        help='number of adversaries to create')


    args = parser.parse_args()
    classes = list(map(lambda x : int(x), args.classes))

    run(args.model_directory, args.model_name, args.adversary_directory, classes=classes, p=args.probability, upto=args.up_to)


if __name__ == '__main__':
    main()
