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

import torchvision.models as models
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
Simulates a step of the environment. Returns the new state, and the next dynamic action space.
Returns a next state: 1x10 vector

'''
def environment_step(train_dataloader, solver, task_model, num_repeats=1):

    accs = torch.zeros((num_repeats,1))
    class_accs_across_runs = torch.zeros((num_repeats,5))
    counts_across_runs = torch.zeros((num_repeats,5))

    for i in range(num_repeats):
        task_model = model.FCNet(num_classes=args.num_classes)  # remake a new task model each time

        acc, vae, discriminator, class_accs = solver.train_without_adv_vae(train_dataloader,
                                                               task_model,
                                                               None,
                                                               None,
                                                               None, args)
        counts, total = dataloader_statistics(train_dataloader, 5)

        counts_across_runs[i] = counts.t()
        accs[i] = acc
        class_accs_across_runs[i] = class_accs

    mean_class_accs = torch.mean(class_accs_across_runs, axis=0).unsqueeze(1)
    mean_class_counts = torch.mean(counts_across_runs, axis=0).unsqueeze(1)

    next_state_vector = torch.cat((mean_class_accs , mean_class_counts), axis=0)

    return torch.mean(accs), next_state_vector.t()

'''simply gets the reward as the acc'''
def compute_reward_clean(curr_state):
    curr_state = curr_state[:,0:args.num_classes].detach()

    return torch.mean(curr_state)

def compute_reward_clean_smoothed(curr_state, time_step, prev_reward):
    curr_state = curr_state[:,0:args.num_classes].detach() #the rward should give 5 signals!
    curr_reward = torch.mean(curr_state) - torch.mean(prev_reward) -(20+time_step*2)
    prev_reward[time_step%len(prev_reward)]  = torch.mean(curr_state)


    return curr_reward, prev_reward

'''
returns the delta, as well as the actual performance
'''
def compute_reward(curr_state, time_step, prev_reward,args):
    curr_state = curr_state[:,0:args.num_classes].detach() #the rward should give 5 signals!
    curr_reward = torch.mean(curr_state) - torch.mean(prev_reward)
    prev_reward[time_step%len(prev_reward)]  = torch.mean(curr_state)


    return curr_reward, prev_reward

    class_acc = curr_state[:,0:5].detach()
    baseline = 20 + 0.8*time_step
    # baseline = 1 + 2*time_step
    # choices: we can try to achieve parity. Or we can try and just maximize the total acc across everything

    # return torch.sum(curr_state-baseline) #equiv to acc.

    # try some torch sum stuff. sum of squared differences for instance
    perf = (class_acc-baseline)
    print("perf is")
    print(torch.mean(perf))
    # return -torch.sum((class_acc - baseline)**2) # we want to achieve 20% acc in all of them...

    return torch.mean(perf)

    # construct a sign vector


    return torch.sum(torch.clamp((class_acc - baseline),min=0, max=100) ** 2)

    print("current performance is ")
    print(perf)

    perf_matrix = perf - perf.t() #broadcasting
    return torch.mean(perf_matrix.pow(2))/2 # compute the sum across, and divide by 2


    # sum along the diagonal etc. perf_matrix


    torch.sum()


    # trying sum of squared errors
    return torch.sum((class_acc - baseline)**2) # we want to achieve 20% acc in all of them...

'''
Penalty for mode collapse. 

Inputs: p and q are the distributions.
'''
def mode_collapse_penalty(p_dist,q_dist):
    return torch.sum((p_dist-q_dist).pow(2))

'''
KL terms
'''

def mode_collapse_penalty_kl(p_dist,q_dist):
    import torch.nn.functional as F
    # p_dist += 0.05
    # # smooth it
    # p_dist /=torch.sum(p_dist)
    
    p_EPSILON = 1e-10

    print(p_dist, q_dist)
    if len((p_dist==0).nonzero()) > 0:
        print("WE HAVE A ZERO")
        print(p_dist)
        p_dist = p_dist + p_EPSILON
        p_dist = p_dist/torch.sum(p_dist)
        print("renormalized")
        print(p_dist)
        
    print("kl term is")
    kl = F.kl_div(p_dist.log(), q_dist, reduction="batchmean") #reverse KL
    print(kl )
    return kl

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
    # data is [features, label, idx, pseudo_label]
    # print("these are some action zero statistics")
    # print(action.probs)
    # print((action.probs<0).nonzero())

    rand = False
    if torch.rand(size=()) < args.epsilon:
        print("randomly chosen")
        rand = True
        rand_idx = torch.randint(len(unlabelled_data), size=())
        datapoint = (unlabelled_data[rand_idx, 0:2], unlabelled_data[rand_idx, 2], unlabelled_data[rand_idx, 3])
        unlabelled_data = np.delete(unlabelled_data, rand_idx, 0)  # test to make sure this works

        targ_cluster = torch.tensor(unlabelled_data[rand_idx][-1],dtype=torch.long).view(-1)
        return targ_cluster, datapoint, unlabelled_data,rand

    targ_cluster = action.sample()


    iters = 0
    datapoint = None
    while iters < 100:
        rand_idx = torch.randint(len(unlabelled_data), size=())

        # we assume the kmeans is appended right at the very end
        if unlabelled_data[rand_idx][-1] == targ_cluster:
            datapoint = (unlabelled_data[rand_idx,0:2], unlabelled_data[rand_idx,2], unlabelled_data[rand_idx,3])
            unlabelled_data = np.delete(unlabelled_data, rand_idx, 0 ) # test to make sure this works
            break

        iters+=1

    return targ_cluster, datapoint, unlabelled_data, rand
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

        new_datapoints_batch = np.zeros((0,3))

        for datapoint_batch, label_batch, _ in  inquiry_dataloader:
            train_ex_batch = np.concatenate((datapoint_batch, np.expand_dims(label_batch, axis=1)), axis=1)
            new_datapoints_batch = np.concatenate((new_datapoints_batch, train_ex_batch), axis=0)  # concat the



        print(sampled_indices)

        current_indices = list(current_indices) + list(sampled_indices) #really they just want a set here...

        sampler = data.sampler.SubsetRandomSampler(current_indices)
        train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                batch_size=args.batch_size, drop_last=False)

        with open(os.path.join(args.out_path, "{}_current_accs.txt".format(args.sampling_method)), "a") as acc_file:
            acc_file.write("{} {}\n".format(acc, class_acc))

        visual_labelled_dataset = np.zeros((0, 3))  # each dimension does not require something new!

        for datapoint_batch, label_batch, _ in train_dataloader:  # will be tuple of n by 1
            train_ex_batch = np.concatenate((datapoint_batch, np.expand_dims(label_batch, axis=1)), axis=1)
            visual_labelled_dataset = np.concatenate((visual_labelled_dataset, train_ex_batch), axis=0)  # concat the

        # visualize_training_dataset(i, args.num_classes, visual_labelled_dataset, new_datapoints_batch, args.sampling_method)

    return accuracies


'''
Computes the per-class statistics for the datapoints in the dataloader
'''
def dataloader_statistics(train_dataloader, num_classes):

    per_class = torch.zeros((num_classes, 1))

    for datapoint, label, idx in train_dataloader:
        for dp, lb, _ in zip(datapoint, label, idx):
            per_class[lb] += 1

    return per_class, torch.sum(per_class)

'''
new datapoints is a numpy array, n by D+1 (D for the features, 1 for the class)
'''
def visualize_training_dataset(iteration, num_classes, prev_dataset, new_datapoints, name="rl"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    for cluster in range(num_classes):
        # k_means_data  = unlabelled_dataset[unlabelled_dataset[...,-1]==cluster]
        # fig, ax = plt.subplots()

        k_means_data  = prev_dataset[prev_dataset[:,-1]==cluster]

        ax.scatter(k_means_data  [:,0], k_means_data  [:,1])
        # break

    ax.scatter(new_datapoints[:,0],new_datapoints[:,1], s=100)
    fig.savefig(os.path.join(args.out_path, "{}_viz_{}".format(name,iteration)))

    fig.show()
    plt.close(fig)


'''
function which computes the REINFORCE paramter updates
'''
def learn(y,y_pred, reward, criterion, optim):
    '''assert the loss is indeed the CE loss'''
    print("why is the loss bounded?")
    print("y_pred {}; y {}".format(y_pred,y))


    loss = criterion( y_pred, y) #auto applies the log softmax, so it should be same!
    print("og loss {}".format(loss))
    loss = reward * loss
    loss *= -1

    print("reward_weighted  here:")
    print(loss)

    optim.zero_grad()
    loss.sum().backward()
    optim.step()


def discount_rewards(r):
    GAMMA = 0.9
    discounted_r = torch.zeros(r.size())
    running_add = 0
    for t in reversed(range(len(r))):
        # print("Step: {}".format(t))
        running_add = running_add * GAMMA + r[t]  # we slowly accumulate the reward into all of the steps along the way
        discounted_r[t] = running_add
    # therefore, this has the effect of adding the rewards back into the original actions..

    # print(r)
    # print(discounted_r)
    return discounted_r

'''
Processes the reward, like reward normalization and importantly,
GAMMA DISCOUNTING.
'''
#TODO: use logging instead of copious prints
def process_reward(reward_history):
    print("processed reward; reward is now")

    adv = discount_rewards(reward_history)
    processed_reward_history = adv
    processed_reward_history = (adv - adv.mean()) / (adv.std() + 1e-7)
    print(reward_history)
    print(processed_reward_history)
    return processed_reward_history

def rl_main(args):

    # a full game, where you pick 10 examples
    # args.mine_episodes = 10

    args.episode_length = 10
    args.num_episodes = 100

    args.epsilon = 0.5 # try with full policy. and try with using the full vector to compute a reward. But it really is just a multiple. Unless we specifically penalize assigning 0 counts

    # probably starting with 10 or so points randomly would be very good. but would invalidate past work

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

    elif args.dataset == "mnist":
        print("Using MNIST dataset...")
        test_dataloader = data.DataLoader(
            MNIST(args.data_path),
            batch_size=args.batch_size, drop_last=False
        )

        train_dataset = MNIST(args.data_path)
        print(len(train_dataset))
        args.num_images = 2500
        args.budget = 1  # how many we can label at each round
        args.initial_budget = 1
        args.num_classes = 10


    random.seed(args.torch_manual_seed)
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


    # iterate the train_dataloader, and compute some statistics. And also, increment quantities.



    '''
    FORMULATION1: We will feed in the class_specific accuracies.
    '''
    ROLLING_AVG_LEN = 5
    prev_reward = torch.ones((ROLLING_AVG_LEN,1))
    prev_reward *=20
    print("prev_reward{}".format(prev_reward))

    STATE_SPACE = args.num_classes
    ACTION_SPACE = args.num_classes
    CLASS_DIST_SPACE = args.num_classes

    pol_class_net = PolicyNet(STATE_SPACE + CLASS_DIST_SPACE, ACTION_SPACE ) # gradient, or hessian in the network..; per class accs as well


    args.old_model_path = "/h/johnchen/Desktop/git_stuff/vaal_querying/testing_zero/bs_16/model.pt"

    if os.path.exists(args.old_model_path) and args.use_old:

        pol_class_net.load_state_dict(torch.load(args.old_model_path))
    pol_optimizer = optim.Adam(pol_class_net.parameters(), lr=5e-2)


    curr_state = torch.zeros((1,STATE_SPACE + CLASS_DIST_SPACE)) #only feed it in the past state directly

    import copy

    task_model = model.FCNet(num_classes=args.num_classes)
    # inference_model = task_model
    # inference_model.to(args.device)
    # task_model = vgg.vgg16_bn(num_classes=args.num_classes)

    accuracies = []
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # feel like supporting a desparate cause; might delete later

    entire_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    # ask on SO: multi item getting using pytorch, dataloader

    features, labels, idx = next(iter(entire_loader))
    features = features.numpy()[unlabeled_indices] # we should exactly not be using it as this. Actually, it is OK. we are just saing what is labelled and what is not
    labels = np.expand_dims(labels.numpy()[unlabeled_indices], 1)
    idx =    np.expand_dims(idx.numpy()[unlabeled_indices], 1)
    # X = np.hstack((features,labels ,idx )) #strange that this doesn't work

    # X = np.concatenate((features.reshape(len(features),-1), labels,idx), axis=1)
    X = np.concatenate((features, labels,idx), axis=1)


    from sklearn.cluster import KMeans
    kmeans_obj = KMeans(n_clusters=args.num_classes, random_state=0)  # we can also fit one kmeans at the very start.
    cluster_preds = kmeans_obj.fit_predict(X[:,0:2])

    oracle_clusters = True

    if oracle_clusters:
        unlabelled_dataset = np.concatenate((X, labels), axis=1)

    else:
    # we can also just predict (should be fast) again on new datapoints, using the trained classifier. But why not just memorize
        unlabelled_dataset = np.concatenate((X, np.expand_dims(cluster_preds,axis=1)), axis=1)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()


    for cluster in range(args.num_classes):
        # k_means_data  = unlabelled_dataset[unlabelled_dataset[...,-1]==cluster]
        # fig, ax = plt.subplots()

        k_means_data  = unlabelled_dataset[unlabelled_dataset[:,-1]==cluster]

        ax.scatter(k_means_data  [:,0], k_means_data  [:,1])
        ax.scatter(kmeans_obj.cluster_centers_[cluster][0], kmeans_obj.cluster_centers_[cluster][1], s=100)
        fig.savefig(os.path.join(args.out_path, "cluster_{}".format(cluster)))
        # break

    fig.show()

    # gradient_accum = torch.zeros((args.rl_batch_steps, 1), requires_grad=False) # accumulate all the losses

    # try making it an empty thing
    gradient_accum = torch.zeros((args.episode_length), requires_grad=False) # accumulate all the losses

    # loss.backward(0 => doesn't actually execute an update of the weights. we could probably call loss.backward individually

    batched_accs = []


    # try combining it with the state. and also, just try doing an epsilon greedy policy
    import torch.nn.functional as F
    for i in tqdm(range(args.num_episodes)):

        # need y, y_pred and the reward
        action_history = torch.FloatTensor([])
        reward_history = torch.FloatTensor([])
        taken_action_history = torch.LongTensor([])
        current_indices = [] #reset the current indices
        prev_reward = torch.ones((ROLLING_AVG_LEN, 1))
        prev_reward *= 20
        for j in range(args.episode_length):
            pol_optimizer.zero_grad()

            # here we need a fake label, in order to back prop the loss. And don't backprop immediately, instead, get the gradient,
            # hold it, wait for the reward, and then backprop on that quantity
            action_vector = pol_class_net (curr_state )
            # torch.nn.functional.log_softmax(action_vector)
            # action_dist = torch.zeros((1))

            # the huge bug is as follows:
            # make the network output a softmax
            # then, the loss is the cross entropy:
            # print(F.softmax(action_vector))
            if (len(F.softmax(action_vector)[F.softmax(action_vector)<0]) > 0 ):
                print("huh 0 probs!!")

            action_probs = F.softmax(action_vector)
            action_dist = torch.distributions.Categorical(probs=action_probs) #the diff between Softmax and softmax

            action_history = torch.cat([action_history, action_vector])

            # we probably need logsoftmax here too
            # print("action dist{}\n, dist probs{}\n, "
            #       "self f.softmax {}\n, self.log softmax{}\n".format(action_vector,
            #                                                          action_dist.probs,
            #                                                          F.softmax(action_vector, dim=1),
            #                                                                                       F.log_softmax(action_vector,
            #                                                                                                 dim=1))) #intelligent to take the softmax over the right dimension
            # print() #recall logsoftmax and such

            # if torch.rand() < args.epsilon:
            #     pass
            # else:
            # correct_label1, action1 = get_query(action_dist, unlabeled_dataloader, inference_model, args)
            correct_label, action, unlabelled_dataset, rand = get_query_via_kmeans(action_dist, unlabelled_dataset, args)



            taken_action_history = torch.cat([taken_action_history, torch.LongTensor([(correct_label)])])

            # labelled updates
            current_indices = list(current_indices) + [int(action[2].item())] # really they just want a set here...
            sampler = data.sampler.SubsetRandomSampler(current_indices)

            # dump all the indices now

            train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                                               batch_size=args.batch_size, drop_last=False)

            # unlabelled updates
            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
            unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
            unlabeled_dataloader = data.DataLoader(train_dataset,
                                                   sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

            class_counts, total = dataloader_statistics(train_dataloader, args.num_classes)
            # print("class counts {}".format(class_counts))



            #data loader not subscriptable => we should deal with the indices.
            # we could also combine, and get the uncertainties, but WEIGHTED BY CLASS
            # lets just try the dataloader, but it will be challenging when we have the batch size...
            # print(correct_label)
            # print("this is the action taken by the sampler")
            # print(action)

            acc, curr_state = environment_step(train_dataloader, solver, task_model) #might need to write a bit coupled code. This is OK for now


            reward,prev_reward = compute_reward_clean_smoothed(curr_state,i*0+j, prev_reward) #move the reward to the env. step.
            reward_history = torch.cat([reward_history, reward.unsqueeze(0)])

            accuracies.append(acc)
            print(prev_reward)
            print(action_dist.probs)
            print(F.softmax(action_vector))




        # end of episode; compute the reward and go forth!
        if True:
            processed_reward_history = process_reward(reward_history)
            learn(taken_action_history, action_history , processed_reward_history, criterion, pol_optimizer)

            with open(os.path.join(args.out_path, "accs.txt"), "a") as acc_file:
                acc_file.write("{};{}\n".format(acc, curr_state))


            print(curr_state)
            print(acc)

            # with open(os.path.join(args.out_path, "rl_current_accs.txt"), "a") as acc_file:
            #     acc_file.write("{} {}\n".format(acc, class_accs))

            # inference_model = task_model
            # inference_model.to(args.device)
            task_model = model.FCNet(num_classes=args.num_classes) # remake a new task model each time
            # task_model = vgg.vgg16_bn(num_classes=args.num_classes)

            # graph the train dataloader at each iteration



            # for cluster in range(args.num_classes):
            #     # k_means_data  = unlabelled_dataset[unlabelled_dataset[...,-1]==cluster]
            #     # fig, ax = plt.subplots()
            #
            #     k_means_data = unlabelled_dataset[unlabelled_dataset[:, -1] == cluster]
            #
            #     ax.scatter(k_means_data[:, 0], k_means_data[:, 1])
            #     ax.scatter(kmeans_obj.cluster_centers_[cluster][0], kmeans_obj.cluster_centers_[cluster][1], s=100)
            #     fig.savefig(os.path.join(args.out_path, "cluster_{}".format(cluster)))
            visual_labelled_dataset = np.zeros((0,3)) #each dimension does not require something new!

            new_datapoints= np.reshape(np.asarray(action[0]), newshape=(-1,2))
            for datapoint_batch, label_batch, _ in train_dataloader: #will be tuple of n by 1
                train_ex_batch = np.concatenate((datapoint_batch, np.expand_dims(label_batch,axis=1)), axis=1)
                visual_labelled_dataset = np.concatenate((visual_labelled_dataset, train_ex_batch), axis=0 ) #concat the

        # visualize_training_dataset(i, args.num_classes, visual_labelled_dataset, new_datapoints)
            #     stack all of them!
            # and furthermore, we need to do a group by on the label.

        #     TODO: reset the batched points, and then readd everything back in

        # i.e.: we need to reset the state, and then do another 10-point learning

        # now, check the visual labelled dataset


        # let's graph the vector, as we see it come
        # graph the new point on the map, then graph the old collection of data as regular

        # current_indices


    # save the trained model
    model_params = pol_class_net.state_dict()
    torch.save(model_params, os.path.join(args.out_path, "model.pt"))



    #

    fig, ax = acc_plot(accuracies, args, label="policy gradient", name="policy gradient only")

    spaced_x = list(range(len(batched_accs)))
    spaced_x = [x*10 for x in spaced_x]
    ax.plot(spaced_x, batched_accs, marker="x", c="purple", label="batched policy updates")
    ax.legend()
    fig.show()
    fig.savefig(os.path.join(args.out_path, "comparison_batched_acc_plot_{}_queries".format(len(accuracies))))


    print(pol_class_net)

    import copy
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

    with open(os.path.join(args.out_path, "args.txt"), "w") as file:

        for key,val in vars(args).items():
            file.write("{}:{}\n".format(key,val))

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
    splits = range(int(math.ceil(100/args.budget))+1) #add 1 for extra batch size

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

        class_counts , total = dataloader_statistics(train_dataloader, 5)

        with open(os.path.join(args.out_path, "class_counts.txt"), "a") as file:
            file.write("{} {}\n".format(class_counts, total))

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
    torch.manual_seed(int(args.torch_manual_seed))

    args.device = None
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    args.gen_plots = False
    args.out_path = os.path.join(args.out_path, args.log_name)
    args.use_old = False
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    if args.gen_plots:

        # this is a target for parallelization
        import torch
        root_dir = args.out_path
        for i in range(0, 30):
            rand_run = torch.randint(high=1000000, size=())
            # args.log_name = "kl_penalty_{}".format(i)
            args.torch_manual_seed = rand_run
            subdir = "kl_penalty_{}".format(i)

            print("this is out path")
            print(args.out_path)
            args.out_path = os.path.join(root_dir, subdir)
            
            if not os.path.exists(args.out_path):
                os.mkdir(args.out_path)
            # main(args)



            rl_main(args)
    else:

        rl_main(args)