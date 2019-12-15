import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from solver import Solver
from utils import *
import arguments


def cifar_transformer():
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

def main(args):


    with open(os.path.join(args.out_path, "args.txt"), "w") as file:

        for key,val in vars(args).items():
            file.write("{}:{}\n".format(key,val))





    if args.dataset == "ring":
        print("Using Ring dataset...")
        test_dataloader = data.DataLoader(
            Ring(args.data_path, transform=simple_data_transformer(), return_idx=False, testset=True),
            batch_size=args.batch_size, drop_last=False
        )

        train_dataset = Ring(args.data_path, simple_data_transformer())
        print(len(train_dataset))
        args.num_images = 2500
        args.budget = 1 #how many we can label at each round
        args.initial_budget = 1
        args.num_classes = 5 
    
    elif args.dataset == 'mnist':
        test_dataloader = data.DataLoader(
                datasets.MNIST(args.path, download=True, transform=mnist_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = MNIST(args.data_path)
        print(len(train_dataset))
        args.num_images = 6000
        args.budget = 300
        args.initial_budget = 300
        args.num_classes = 10

    elif args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)

        args.num_images = 5000
        args.budget = 250
        args.initial_budget = 500
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100

    elif args.dataset == 'imagenet':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset = ImageNet(args.data_path)

        args.num_images = 1281167
        args.budget = 64060
        args.initial_budget = 128120
        args.num_classes = 1000
    else:
        raise NotImplementedError

    random.seed("csc2547")

    all_indices = set(np.arange(args.num_images))
    initial_indices = random.sample(all_indices, args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)

    # dataset with labels available
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
            batch_size=args.batch_size, drop_last=False)

    print(len(train_dataloader))
    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)
    import math
    splits = range(int(math.ceil(100/args.budget)))

    # # splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # splits = [args.initial_budget/float(args.num_images),
    #     (args.initial_budget+args.budget)/float(args.num_images),
    #     (args.initial_budget+args.budget*2)/float(args.num_images),
    #     (args.initial_budget+args.budget*3)/float(args.num_images),
    #     (args.initial_budget+args.budget*4)/float(args.num_images),
    #     (args.initial_budget+args.budget*5)/float(args.num_images), ]

    current_indices = list(initial_indices)
    accuracies = []

    best_data_point = None
    total_optimal =0
    for split in splits:
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        task_model = model.FCNet(num_classes=args.num_classes)
        if args.dataset == "mnist":
            vae = model.VAE(args.latent_dim, nc=1)
        elif args.dataset == "ring":
            vae = model.VAE(args.latent_dim, nc=2)
        else:
            vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim)


        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)


        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)
        print(len(unlabeled_dataloader))

        # print("currently:")
        # print(len(current_indices))
        # print(current_indices)
        #
        # print(len(unlabeled_dataloader))
        #
        # print(len(train_dataset)) # this will always be 2500. but the sampler is an ultimate funneller. But what types of indices do we get?
        # # A: they are scaled to the sampler. But the internal indices? They are similarly scaled.
        #
        # for i in unlabeled_dataloader:
        #     print(i[2]) # they keep their identity. (they have the same index; this is not changed thankfully.) Hence, we can find the uncertainties, and then

        # as the sampler changes, so too does the dataloader
        # sampled_indices = random.choice(unlabeled_indices)
        # current_indices = list(current_indices) + [sampled_indices] #really they just want a set here...


        # continue
        # unlabeled_dataloader = data.DataLoader(train_dataset,
        #                                        sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)


        if args.sampling_method == "adversary" or args.sampling_method == "adversary_1c":
            # train the models on the current data
            # we also want to check which sampled_indice is best, and which one should be ideal, according to the dataset!
            acc, vae, discriminator = solver.train(train_dataloader,
                                                task_model, 
                                                vae, 
                                                discriminator,
                                                unlabeled_dataloader)
        else:
            # train the models on the current data
            acc, vae, discriminator = solver.train_without_adv_vae(train_dataloader,
                                                task_model, 
                                                vae, 
                                                discriminator,
                                                unlabeled_dataloader)


        print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
        accuracies.append(acc)


        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader, task_model)

        args.oracle_impute = False


        if args.oracle_impute:
            print("main")
            print(sampled_indices)
            # compute a pass over all points
            uncertainties =[]
            index_order = [None for _ in range(args.num_images)]



            # torch.ones((args.num_images))

                # [1 for i in range(args.num_images)] # we have so many points to label, in total. we should investigate that if we select a number than, the length is unchanged?

            # get the length. But also get the maximum element. Most likely, the length of the unlabelled dataloader does not change.

            with torch.no_grad():
                for i,pt in enumerate(unlabeled_dataloader):

                    pred = task_model(pt[0].to(args.device))
                    if pred.max().item() < 0.2:
                        print("WHAT")
                        print(pred)
                    uncertainties.append(1 - pred.max().item() )
                    index_order[pt[2].item()] = i

                    # uncertainties[pt[2].item()] = 1 - pred.max().item() # we need to compute the loss! (or we can just take the max uncertainty..)

            # uncertainties can range from 0 to 1. We want to take the maximum value
            # it might be the case that model is perfectly confident. this means that we will have 0. that is ok. it doesn't make sense if we have 1

            # HOW IS IT POSSIBLE WE HAVE 0 AS THE MAX
            # AND HOW IS IT POSSIBLE WE DONT EVEN HAVE THE SAME SAMPLED INDICE AS WHAT THE OTHER RETURNS
            # uncertainties = [elt for elt in uncertainties if elt is not None]
            # ensure that the uncertainties return here are indeed accurate
            # they wont actually line up unfortunately...
            print("uncertainty vs sampled index")
            print(uncertainties.index(max(uncertainties)),max(uncertainties) )
            print(index_order[sampled_indices[0]], uncertainties[index_order[sampled_indices[0]]]) # but it might be possible, that this quantity is not computed...no. it MUST be computed, since it is unlabelled

            # we only multiply by -1 at the end (to select the elemnts which are furtherst)
            # for i in range(len(uncertainties)):
            #     uncertainties[i] = 1 - uncertainties[i]



            # print(sampled_indices)
            # unlabeled_dataloader.dataset[sampled_indices]
            best_data_point,max_acc, accs = oracle_best_point( unlabeled_dataloader, current_indices.copy(), train_dataset, solver, args, sampled_indices, index_order, uncertainties) #since we might have an elt with index being 2.5k, then it would mess it all up.
            # hence, a hash based approach IS best!
            #
            #
            print(sampled_indices, accs[index_order[sampled_indices.item()]])
            print(best_data_point, max_acc)
            #
            if best_data_point == sampled_indices[0]:
                total_optimal += 1
                print(max_acc) #this should be optimal
                # print()
            torch.save(accs, os.path.join(args.out_path, "accs_{}".format(split) + ".txt"))
            torch.save(uncertainties, os.path.join(args.out_path, "uncertainties_{}".format(split) + ".txt"))
            uncertainty_acc_plot(uncertainties, accs, args, split, sampled_indices, index_order)

        #
        query_analysis(sampled_indices, unlabeled_dataloader, args, split)



        # current_indices = list(current_indices) + [best_data_point] #really they just want a set here...
        current_indices = list(current_indices) + list(sampled_indices) #really they just want a set here...

        sampler = data.sampler.SubsetRandomSampler(current_indices)
        train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                batch_size=args.batch_size, drop_last=False)




        # break

    acc_plot(accuracies, args)
    print("In total, we had {} out of 100 optimal".format(total_optimal))

    torch.save(accuracies, os.path.join(args.out_path, args.log_name + ".txt"))


#     do the indices also get readjusted? doesn't appear so...


#
#
# def write_results(filename, ):
#
#
#     pass

'''
computes (via brute force) the best point that SHOULD have been selected to best reduce the gradient loss (or acc)
Assumes the dataloader is batch size 1
'''
def oracle_best_point( unlabeled_dataloader, orig_indices, train_dataset , solver, args, sampled_indices, index_order, uncertainties):

    args.oracle_train_iterations = 200
    print("uncertainty sampling determined {} was best".format(sampled_indices) )

    # uncertainties = [0 for i in range(2500)]#2500 total datapoints
    accs = [None for i in range(len(uncertainties))]
    # indices_order = []

    max_acc = -1
    max_acc_datapoint = None
    # curr_acc = 1
    from tqdm import tqdm
    # we really want to try keeping the training from what we already have...

    for datapoint in tqdm(unlabeled_dataloader): # this should get smaller each time
        task_model = model.FCNet(num_classes=args.num_classes)
        if args.dataset == "mnist":
            vae = model.VAE(args.latent_dim, nc=1)
        elif args.dataset == "ring":
            vae = model.VAE(args.latent_dim, nc=2)
        else:
            vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim)


        # we should get the exact index of the datapoint
        current_indices = list(orig_indices) + [(datapoint[2].item())]  # really they just want a set here...
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        next_train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                           batch_size=args.batch_size, drop_last=False)

        acc, vae, discriminator = solver.oracle_train_without_adv_vae(next_train_dataloader,
                                                               task_model,
                                                               vae,
                                                               discriminator,
                                                               unlabeled_dataloader)

        accs[index_order[datapoint[2].item()]] = acc

        print(datapoint[2].item(), acc)
        if acc > max_acc:
            max_acc = acc
            max_acc_datapoint = datapoint[2].item()

        elif acc == max_acc and sampled_indices[0] == datapoint[2].item():
            max_acc_datapoint = datapoint[2].item() #should take care of the stuff! in general, we do draw the general conclusion that uncertainty might not be best!

    # accs = [elt for elt in accs if elt is not None]
    return max_acc_datapoint, max_acc, accs #in the future we can take max and the index of the max



def acc_plot(accs, args):
    import matplotlib.pyplot as plt
    fig,ax =plt.subplots()
    ax.plot(range(0, len(accs)), accs, marker="x")
    ax.set_title("Accuracy vs query; {} train iterations".format(args.train_iterations))
    ax.set_ylabel("accuracy")
    ax.set_xlabel("queries")

    fig.show()
    fig.savefig(os.path.join(args.out_path,"acc_plot_{}_queries".format(len(accs))))


def uncertainty_acc_plot(uncertainties, accs, args, split,sampled_indices, index_order):
    import matplotlib.pyplot as plt
    fig,ax =plt.subplots()

    # we should filter out accs and uncertainties where they are already selected...

    ax.scatter(uncertainties, accs, marker="x")

    ax.set_title("Accuracy vs Uncertainty sampling; {} train iterations".format(args.train_iterations))
    ax.set_ylabel("accuracy")
    ax.set_xlabel("uncertainties")

    max_ind = accs.index(max(accs))
    # re draw the extremum in better colours
    ax.scatter(uncertainties[index_order[sampled_indices[0]]], accs[index_order[sampled_indices[0]]],c="red", marker="o", label="uncertainty sample", s=64) #inconsistency in matplotlib documentation

    ax.scatter(uncertainties[max_ind], accs[max_ind ], c="green", marker="^", label="optimum sample",s=64)
    ax.legend()

    fig.show()
    fig.savefig(os.path.join(args.out_path,"acc_uncertainty_{}_plot".format(split)))

def query_analysis(queried_indices, unlabeled_dataloader, args, split):

    # args.num_classes
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()


    # we just want the class label of all these
    labels = []

    import collections


    queried_indices = np.asarray(queried_indices).reshape(1,-1)[0,:]
    for x in queried_indices:
        print(unlabeled_dataloader.dataset[x])  # (X, class label, index)
        labels.append(int(unlabeled_dataloader.dataset[x][1]))
    bar_freqs = collections.Counter(labels)
    # get the elements of the list, according to these indices we give you

    print(sorted(bar_freqs.items())) # by value



    # we can sort the bar freqs too
    # does keys() correspond to values()? it might!

    # ax.hist(labels, bins=[i for i in range(5)])

    ax.bar(list(bar_freqs.keys()), list(bar_freqs.values()), align='center')
    ax.set_xticks(list(bar_freqs.keys()))

    # ax.bar(bar_freqs)
    # let's use a python counter

    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_title("Histogram of query")

    fig.show()
    fig.savefig(os.path.join(args.out_path,"query_hist_{}.png".format(split)))


if __name__ == '__main__':
    args = arguments.get_args()

    args.device = None
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main(args)

