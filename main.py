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
import torch.optim as optim

from rl.PolicyNetwork import PolicyNet
from tqdm import tqdm

from identity_sampler import IdentitySampler
def cifar_transformer():
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

'''
Simulates a step of the environment. Returns the new state, and the next dynamic action space

'''
def environment_step(train_dataloader, solver, task_model):
    acc, vae, discriminator, class_accs = solver.train_without_adv_vae(train_dataloader,
                                                           task_model,
                                                           None,
                                                           None,
                                                           None, args)

    return acc, class_accs.unsqueeze(0)

def compute_reward(curr_state, time_step):
    class_acc = curr_state
    baseline = 20 + 0.8*time_step

    # choices: we can try to achieve parity. Or we can try and just maximize the total acc across everything

    return torch.sum(curr_state-baseline) #equiv to acc.
    # trying sum of squared errors
    return torch.sum((class_acc - baseline)**2) # we want to achieve 20% acc in all of them...


'''
Returns the actual query, given an action distribution.
# just use the task_model as a noisy predictor. Take 10 samples, and then take the one that is predicted as most likely to 
# be reported as the correct sample

return a tupel: target class (action selected) and datapoint tuple (x[best_idx], _[best_idx], idx[best_idx])
'''
def get_query(action, unlabelled_dataloader, task_model, args):

    targ_class = action.sample()

    # now, we just need to sample the class

    datapoint = None
    # from the unlabelled indices, sample an appropriate point from the class
    iters = 0

    lowest_score = 100
    lowest_datapoint = None
    # print(len(unlabelled_dataset))

    for batch in unlabelled_dataloader:
        x, _, idx = batch

        preds = task_model(x.to(args.device))

        # best_idx = torch.argmin(torch.abs(preds[:,targ_class] -0.5))
        best_idx = torch.argmin((preds[:,targ_class]))

        return targ_class, (x[best_idx], _[best_idx], idx[best_idx])
        # break


    # return targ_class, lowest_datapoint

    # pass

'''
Gets query using k-means clustering. We assume a k-means clustering is passed in

Meta note: this is often done: we have a function that needs to "continue" its work: this is continuation, generator, yield etc.

args:
    - kmeans: a kmeans objects
    - action: the action label
    - unlabelled_data: a NUMPY array of the full dataset (x,y, class and label). We will then hstack or something. And also make sure to delete the row when we sample
        - should be of size 3 X ?
        - in reality, should be of size (N X [sum (feature_space) +1]). i.e. we will squahs all the features across. 
         
    - unlabelled_mask: a boolean mask that specifies whether a point has already been labelled or not 
    - ideally, i would like to do kmeans clustering on only PART of the array. Since, i can get the x,y coord of the 
    - data points, but then i want to convert BACK to 
    - what the actual indices are!.. yes we should be able to do kmeans on a numpy array. And then have the cluster appear as another feature 
    of data points that 
'''

def get_query_via_kmeans(action, unlabelled_data, args):

    targ_class = action.sample()


    iters = 0
    datapoint = None
    while iters < 100:
        rand_idx = torch.randint(len(unlabelled_data), size=())

        # we assume the kmeans is appended right at the very end
        if unlabelled_data[rand_idx][-1] == targ_class:
            datapoint = (unlabelled_data[rand_idx,0:2], unlabelled_data[rand_idx,2], unlabelled_data[rand_idx,3])
            unlabelled_data = np.delete(unlabelled_data, rand_idx, 0 ) # test to make sure this works
            break

        iters+=1

    return targ_class, datapoint, unlabelled_data
    # now, we just keep track of the indices
    # now, we just need to sample the class

    datapoint = None
    # from the unlabelled indices, sample an appropriate point from the class
    iters = 0

    lowest_score = 100
    lowest_datapoint = None
    # print(len(unlabelled_dataset))


def random_baseline(args, num_iters=100):



    # runs a random baseline


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



    solver = Solver(args, test_dataloader)

    all_indices = set(np.arange(args.num_images))
    initial_indices = random.sample(all_indices, args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    current_indices = list(initial_indices)

    unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)

    # dataset with labels available
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                       batch_size=args.batch_size, drop_last=False)
    accuracies = []

    for i in range(num_iters):
        task_model = model.FCNet(num_classes=args.num_classes) # remake a new task model each time

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)


        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)


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
            acc,_,_,class_acc = solver.train_without_adv_vae(train_dataloader,
                                                task_model,
                                                None,
                                                None,
                                                None, args)




        print('Final accuracy with {}% of data is: {:.2f}'.format(int(i), acc))
        print(class_acc)
        accuracies.append(acc)


        sampled_indices = solver.sample_for_labeling(None, None, unlabeled_dataloader, task_model)


        inquiry_sampler = data.sampler.SubsetRandomSampler(sampled_indices)

        inquiry_dataloader = data.DataLoader(train_dataset, sampler=inquiry_sampler ,
                batch_size=args.batch_size, drop_last=False)

        for elt in inquiry_dataloader:
            print(elt)


        print(sampled_indices)

        current_indices = list(current_indices) + list(sampled_indices) #really they just want a set here...

        sampler = data.sampler.SubsetRandomSampler(current_indices)
        train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                batch_size=args.batch_size, drop_last=False)

        with open(os.path.join(args.out_path, "{}_current_accs.txt".format(args.sampling_method)), "a") as acc_file:
            acc_file.write("{} {}\n".format(acc, class_acc))



    return accuracies


def rl_main(args):
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

    random.seed(2547)
    torch.manual_seed(args.torch_manual_seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    solver = Solver(args, test_dataloader)

    all_indices = set(np.arange(args.num_images))
    initial_indices = random.sample(all_indices, args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    current_indices = list(initial_indices)

    unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
    unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
    unlabeled_dataloader = data.DataLoader(train_dataset,
                                           sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

    # dataset with labels available
    train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                       batch_size=args.batch_size, drop_last=False)

    '''
    FORMULATION1: We will feed in the class_specific accuracies.
    '''
    STATE_SPACE = args.num_classes
    ACTION_SPACE = args.num_classes


    pol_class_net = PolicyNet(STATE_SPACE , ACTION_SPACE ) # gradient, or hessian in the network..; per class accs as well
    pol_optimizer = optim.Adam(pol_class_net.parameters(), lr=5e-3)
    args.num_episodes = 100


    curr_state = torch.zeros((1,STATE_SPACE)) #only feed it in the past state directly

    import copy

    task_model = model.FCNet(num_classes=args.num_classes)
    inference_model = task_model
    inference_model.to(args.device)

    accuracies = []
    criterion = torch.nn.CrossEntropyLoss()

    # feel like supporting a desparate cause; might delete later

    entire_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

    # ask on SO: multi item getting using pytorch, dataloader

    features, labels, idx = next(iter(entire_loader))
    features = features.numpy()[unlabeled_indices] # we should exactly not be using it as this. Actually, it is OK. we are just saing what is labelled and what is not
    labels = np.expand_dims(labels.numpy()[unlabeled_indices], 1)
    idx =    np.expand_dims(idx.numpy()[unlabeled_indices], 1)
    # X = np.hstack((features,labels ,idx )) #strange that this doesn't work

    X = np.concatenate((features, labels,idx), axis=1)


    from sklearn.cluster import KMeans
    cluster_preds = KMeans(n_clusters=ACTION_SPACE, random_state=0).fit_predict(X)  # we can also fit one kmeans at the very start.
    # we can also just predict (should be fast) again on new datapoints, using the trained classifier. But why not just memorize
    unlabelled_dataset = np.concatenate((X, np.expand_dims(cluster_preds,axis=1)), axis=1)
    for i in tqdm(range(args.num_episodes)):
        pol_optimizer.zero_grad()

        # here we need a fake label, in order to back prop the loss. And don't backprop immediately, instead, get the gradient,
        # hold it, wait for the reward, and then backprop on that quantity
        action_vector = pol_class_net (curr_state )

        action_dist = torch.distributions.Categorical(torch.nn.functional.softmax(action_vector)) #the diff between Softmax and softmax
        print(action_dist.probs)

        # correct_label1, action1 = get_query(action_dist, unlabeled_dataloader, inference_model, args)
        correct_label, action, unlabelled_dataset = get_query_via_kmeans(action_dist, unlabelled_dataset, args)


        pred_vector = action_vector.view(1,-1)
        correct_label = correct_label
        loss =criterion (pred_vector, correct_label)



        # labelled updates
        current_indices = list(current_indices) + [int(action[2].item())] # really they just want a set here...
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                           batch_size=args.batch_size, drop_last=False)

        # unlabelled updates
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                                               sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        #data loader not subscriptable => we should deal with the indices.
        # we could also combine, and get the uncertainties, but WEIGHTED BY CLASS
        # lets just try the dataloader, but it will be challenging when we have the batch size...
        # print(correct_label)
        print(action)

        acc, curr_state = environment_step(train_dataloader, solver, task_model) #might need to write a bit coupled code. This is OK for now
        accuracies.append(acc)

        # if i % args.rl_batch==0:
        #     torch.stack()
        #     compute the reward. store the gradients
        # store all the gradients, then torch.mean them, and then take a step. This means we only have 10/50 steps.

        reward = compute_reward(curr_state, i) # basline is around 1% improvement
        loss *= reward
        loss.backward()
        pol_optimizer.step()


        print(curr_state)
        print(acc)

        with open(os.path.join(args.out_path, "rl_current_accs.txt"), "a") as acc_file:
            acc_file.write("{} {}\n".format(acc, curr_state))

        inference_model = task_model
        inference_model.to(args.device)
        task_model = model.FCNet(num_classes=args.num_classes) # remake a new task model each time

    acc_plot(accuracies, args, label="policy gradient", name="policy gradient only")

    print(pol_class_net)

    uncertain_args = copy.deepcopy(args)
    uncertain_args.sampling_method = "uncertainty"
    uncertain_accs = random_baseline(uncertain_args, args.num_episodes)

    random_args = copy.deepcopy(args)
    random_args.sampling_method = "random"
    random_accs = random_baseline(random_args, args.num_episodes)

    fig, ax =    acc_plot(accuracies, args, label="policy gradient")
    ax.plot(range(0, len(random_accs)), random_accs, marker="x", c="orange", label="random")
    ax.plot(range(0, len(uncertain_accs)), uncertain_accs, marker="^", c="green", label="uncertain")


    ax.legend()
    fig.show()
    fig.savefig(os.path.join(args.out_path, "comparison_acc_plot_{}_queries".format(len(accuracies))))



    # try comparing vs a regularly trained network (random sampling)

    # now that the policy network is trained, we can use it to actually do the inference

    # #     ever batch size episodes, make the gradient update
    # #     compute a rollout, or at least one step
    # # need an environment step
    #
    #
    # print(len(train_dataloader))
    #
    #
    # import math
    # splits = range(int(math.ceil(100 / args.budget)))
    #
    # # # splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # # splits = [args.initial_budget/float(args.num_images),
    # #     (args.initial_budget+args.budget)/float(args.num_images),
    # #     (args.initial_budget+args.budget*2)/float(args.num_images),
    # #     (args.initial_budget+args.budget*3)/float(args.num_images),
    # #     (args.initial_budget+args.budget*4)/float(args.num_images),
    # #     (args.initial_budget+args.budget*5)/float(args.num_images), ]
    #
    # current_indices = list(initial_indices)
    # accuracies = []
    #
    # best_data_point = None
    # total_optimal = 0
    #
    # for split in splits:
    #     task_model = model.FCNet(num_classes=args.num_classes)  # remake a new task model each time
    #     # need to retrain all the models on the new images
    #     # re initialize and retrain the models
    #     # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
    #     if args.dataset == "mnist":
    #         vae = model.VAE(args.latent_dim, nc=1)
    #     elif args.dataset == "ring":
    #         vae = model.VAE(args.latent_dim, nc=2)
    #     else:
    #         vae = model.VAE(args.latent_dim)
    #     discriminator = model.Discriminator(args.latent_dim)
    #
    #     unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
    #
    #     unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
    #     unlabeled_dataloader = data.DataLoader(train_dataset,
    #                                            sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)
    #     print(len(unlabeled_dataloader))
    #
    #     if args.sampling_method == "adversary" or args.sampling_method == "adversary_1c":
    #         # train the models on the current data
    #         # we also want to check which sampled_indice is best, and which one should be ideal, according to the dataset!
    #         acc, vae, discriminator = solver.train(train_dataloader,
    #                                                task_model,
    #                                                vae,
    #                                                discriminator,
    #                                                unlabeled_dataloader)
    #     else:
    #         # train the models on the current data
    #         acc, vae, discriminator = solver.train_without_adv_vae(train_dataloader,
    #                                                                task_model,
    #                                                                vae,
    #                                                                discriminator,
    #                                                                unlabeled_dataloader)
    #
    #     print('Final accuracy with {}% of data is: {:.2f}'.format(int(split * 100), acc))
    #     accuracies.append(acc)
    #
    #     sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader, task_model)
    #
    #     args.oracle_impute = False
    #     print("ORacle impute: {}".format(args.oracle_impute))
    #
    #
    #
    #     if args.oracle_impute:
    #         print("main")
    #         print(sampled_indices)
    #         # compute a pass over all points
    #         uncertainties = []
    #         index_order = [None for _ in range(args.num_images)]
    #
    #         # torch.ones((args.num_images))
    #
    #         # [1 for i in range(args.num_images)] # we have so many points to label, in total. we should investigate that if we select a number than, the length is unchanged?
    #
    #         # get the length. But also get the maximum element. Most likely, the length of the unlabelled dataloader does not change.
    #
    #         with torch.no_grad():
    #             for i, pt in enumerate(unlabeled_dataloader):
    #                 pred = task_model(pt[0].to(args.device))
    #                 uncertainties.append(1 - pred.max().item())
    #                 index_order[pt[2].item()] = i
    #
    #                 # uncertainties[pt[2].item()] = 1 - pred.max().item() # we need to compute the loss! (or we can just take the max uncertainty..)
    #
    #         # uncertainties can range from 0 to 1. We want to take the maximum value
    #         # it might be the case that model is perfectly confident. this means that we will have 0. that is ok. it doesn't make sense if we have 1
    #
    #         # HOW IS IT POSSIBLE WE HAVE 0 AS THE MAX
    #         # AND HOW IS IT POSSIBLE WE DONT EVEN HAVE THE SAME SAMPLED INDICE AS WHAT THE OTHER RETURNS
    #         # uncertainties = [elt for elt in uncertainties if elt is not None]
    #         # ensure that the uncertainties return here are indeed accurate
    #         # they wont actually line up unfortunately...
    #         print("uncertainty vs sampled index")
    #         print(uncertainties.index(max(uncertainties)), max(uncertainties))
    #         print(index_order[sampled_indices[0]], uncertainties[index_order[sampled_indices[
    #             0]]])  # but it might be possible, that this quantity is not computed...no. it MUST be computed, since it is unlabelled
    #
    #         # we only multiply by -1 at the end (to select the elemnts which are furtherst)
    #         # for i in range(len(uncertainties)):
    #         #     uncertainties[i] = 1 - uncertainties[i]
    #
    #         # print(sampled_indices)
    #         # unlabeled_dataloader.dataset[sampled_indices]
    #         best_data_point, max_acc, accs = oracle_best_point(unlabeled_dataloader, current_indices.copy(),
    #                                                            train_dataset, solver, args, sampled_indices,
    #                                                            index_order,
    #                                                            uncertainties)  # since we might have an elt with index being 2.5k, then it would mess it all up.
    #         # hence, a hash based approach IS best!
    #         #
    #         #
    #         print(sampled_indices, accs[index_order[sampled_indices.item()]])
    #         print(best_data_point, max_acc)
    #         #
    #         if best_data_point == sampled_indices[0]:
    #             total_optimal += 1
    #             print(max_acc)  # this should be optimal
    #             # print()
    #         torch.save(accs, os.path.join(args.out_path, "accs_{}".format(split) + ".txt"))
    #         torch.save(uncertainties, os.path.join(args.out_path, "uncertainties_{}".format(split) + ".txt"))
    #         uncertainty_acc_plot(uncertainties, accs, args, split, sampled_indices, index_order)
    #
    #     with open(os.path.join(args.out_path, "current_accs.txt"), "a") as acc_file:
    #         acc_file.write("{}\n".format(acc))
    #
    #     #
    #     query_analysis(sampled_indices, unlabeled_dataloader, args, split)
    #
    #     # current_indices = list(current_indices) + [best_data_point] #really they just want a set here...
    #     current_indices = list(current_indices) + list(sampled_indices)  # really they just want a set here...
    #
    #     sampler = data.sampler.SubsetRandomSampler(current_indices)
    #     train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
    #                                        batch_size=args.batch_size, drop_last=False)
    #
    #     # break
    #
    # acc_plot(accuracies, args)
    # print("In total, we had {} out of 100 optimal".format(total_optimal))
    #
    # torch.save(accuracies, os.path.join(args.out_path, args.log_name + ".txt"))

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

    np.random.seed(2547)
    random.seed(2547)
    torch.manual_seed(args.torch_manual_seed)

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
    task_model = model.FCNet(num_classes=args.num_classes)

    for split in splits:
        task_model = model.FCNet(num_classes=args.num_classes) # remake a new task model each time
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
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
            acc, vae, discriminator,class_acc = solver.train_without_adv_vae(train_dataloader,
                                                task_model, 
                                                vae, 
                                                discriminator,
                                                unlabeled_dataloader, args)


        print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100), acc))
        accuracies.append(acc)

        print(class_acc)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader, task_model)

        args.oracle_impute = False
        print("ORacle impute: {}".format(args.oracle_impute))


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

        with open(os.path.join(args.out_path, "current_accs.txt"), "a") as acc_file:
            acc_file.write("{}\n".format(acc))




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



# def multi_acc_plot(multi_accs, args, multi_labels):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#
#     for i,accs in enumerate(multi_accs):
#         ax.plot(range(0, len(accs)), accs, label=multi_labels[i])
#
#     ax.set_title("Accuracy vs query; {} train iterations".format(args.train_iterations))
#     ax.set_ylabel("accuracy")
#     ax.set_xlabel("queries")
#
#     fig.show()
#     fig.savefig(os.path.join(args.out_path, "multi_acc_plot_{}_queries".format(len(accs))))
#     return fig, ax



def acc_plot(accs, args, label="uncertainty", name="acc_plot"):
    import matplotlib.pyplot as plt

    file_name = os.path.join(args.out_path, "{}_{}_queries".format(name, len(accs)))
    fig,ax =plt.subplots()
    ax.plot(range(0, len(accs)), accs, marker="o", label=label)
    ax.set_title("Accuracy vs query; {} train iterations".format(args.train_iterations))
    ax.set_ylabel("accuracy")
    ax.set_xlabel("queries")

    fig.show()
    fig.savefig(file_name )
    return fig, ax

'''
A double acc plot, showing class-based performances.
'''
# def double_class_acc_plot(accs1, accs2, args, label1="uncertainty", label2="uncertainty"):
#     import matplotlib.pyplot as plt
#     fig,ax =plt.subplots()
#     ax.plot(range(0, len(accs)), accs, marker="x", label=label)
#     ax.set_title("Accuracy vs query; {} train iterations".format(args.train_iterations))
#     ax.set_ylabel("accuracy")
#     ax.set_xlabel("queries")
#
#     fig.show()
#     fig.savefig(os.path.join(args.out_path,"acc_plot_{}_queries".format(len(accs))))
#     return fig, ax
#
# '''
# assumes that accs is a 5 by N graph.
# '''
# def single_class_acc_plot(accs1, args, label1="uncertainty"):
#     import matplotlib.pyplot as plt
#     fig,ax =plt.subplots()
#
#     for
#
#     ax.plot(range(0, len(accs)), accs, marker="x", label=label)
#     ax.set_title("Accuracy vs query; {} train iterations".format(args.train_iterations))
#     ax.set_ylabel("accuracy")
#     ax.set_xlabel("queries")
#
#     fig.show()
#     fig.savefig(os.path.join(args.out_path,"acc_plot_{}_queries".format(len(accs))))
#     return fig, ax

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

    # main(args)
    rl_main(args)
