import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
import torchvision.models as models

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from mnist_net import MNISTNet
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
def environment_step(train_dataloader, solver, task_model, args, num_repeats=3 ):

    accs = torch.zeros((num_repeats,1))
    class_accs_across_runs = torch.zeros((num_repeats,args.num_classes))
    counts_across_runs = torch.zeros((num_repeats,args.num_classes))

    for i in range(num_repeats):
        # task_model = model.FCNet(num_classes=args.num_classes)  # remake a new task model each time
        # task_model = models.resnet18(pretrained=True, num_classes=args.num_classes)
        # task_model = vgg.vgg16_bn(pretrained=True)
        task_model = MNISTNet()
        acc, vae, discriminator, class_accs = solver.train_without_adv_vae(train_dataloader,
                                                               task_model,
                                                               None,
                                                               None,
                                                               None, args)
        counts, total = dataloader_statistics(train_dataloader, args.num_classes)

        counts_across_runs[i] = counts.t()
        accs[i] = acc
        class_accs_across_runs[i] = class_accs

    mean_class_accs = torch.mean(class_accs_across_runs, axis=0).unsqueeze(1)
    mean_class_counts = torch.mean(counts_across_runs, axis=0).unsqueeze(1)

    next_state_vector = torch.cat((mean_class_accs , mean_class_counts), axis=0)

    return torch.mean(accs), next_state_vector.t()
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

    print("these are some action zero statistics")
    print(action.probs)
    print((action.probs<0).nonzero())
    print((torch.isnan(action.probs)).nonzero())

    rand = False
    if torch.rand(size=()) < args.epsilon:
        rand = True
        rand_idx = torch.randint(len(unlabelled_data), size=())
        # datapoint = <features, label, index>
        datapoint = (unlabelled_data[rand_idx, 0:args.feature_length], unlabelled_data[rand_idx, args.feature_length], unlabelled_data[rand_idx, args.feature_length+1])
        unlabelled_data = np.delete(unlabelled_data, rand_idx, 0)  # test to make sure this works

        targ_cluster = torch.tensor(unlabelled_data[rand_idx][-1],dtype=torch.long).view(-1)
        return targ_cluster, datapoint, unlabelled_data,rand

    # correct_label, action, unlabelled_dataset, rand
    # action is broken into datapoint, label, and the idx?

    targ_cluster = action.sample()


    iters = 0
    datapoint = None
    while iters < 100:
        rand_idx = torch.randint(len(unlabelled_data), size=())

        # we assume the kmeans is appended right at the very end
        if unlabelled_data[rand_idx][-1] == targ_cluster:
            datapoint = (
            unlabelled_data[rand_idx, 0:args.feature_length], unlabelled_data[rand_idx, args.feature_length],
            unlabelled_data[rand_idx, args.feature_length + 1])

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

    elif args.dataset == "mnist":
        print("Using MNIST dataset...")
        test_dataloader = data.DataLoader(
            MNIST(args.data_path, return_idx=False),
            batch_size=args.batch_size, drop_last=False
        )

        train_dataset = MNIST(args.data_path)
        print(len(train_dataset))
        args.num_images = 2500
        args.budget = 1  # how many we can label at each round
        args.initial_budget = 1
        args.num_classes = 10
        args.task_model = MNISTNet

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
    args.num_repeats = 3
    for i in range(num_iters):

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)


        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset,
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)
        sample_accs = np.zeros((args.num_repeats,1))
        for rerun in range(args.num_repeats):
            task_model = args.task_model()  # remake a new task model each time

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


            sample_accs[rerun] = acc

        print('Final accuracy with {}% of data is: {:.2f}'.format(int(i), np.mean(sample_accs)))
        print(class_acc)
        accuracies.append(np.mean(sample_accs))


        sampled_indices = solver.sample_for_labeling(None, None, unlabeled_dataloader, task_model)


        inquiry_sampler = data.sampler.SubsetRandomSampler(sampled_indices)

        inquiry_dataloader = data.DataLoader(train_dataset, sampler=inquiry_sampler ,
                batch_size=args.batch_size, drop_last=False)

        # this shouldbe args.feature_length + 3 (1 for idx, 1 for label, 1 for k-means too btw)
        # new_datapoints_batch = np.zeros((0,3))
        #
        # for datapoint_batch, label_batch, _ in  inquiry_dataloader:
        #     train_ex_batch = np.concatenate((datapoint_batch, np.expand_dims(label_batch, axis=1)), axis=1)
        #     new_datapoints_batch = np.concatenate((new_datapoints_batch, train_ex_batch), axis=0)  # concat the



        print(sampled_indices)

        current_indices = list(current_indices) + list(sampled_indices) #really they just want a set here...

        sampler = data.sampler.SubsetRandomSampler(current_indices)
        train_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                batch_size=args.batch_size, drop_last=False)

        with open(os.path.join(args.out_path, "{}_current_accs.txt".format(args.sampling_method)), "a") as acc_file:
            acc_file.write("{} {}\n".format(acc, class_acc))

        visual_labelled_dataset = np.zeros((0, 3))  # each dimension does not require something new!

        # for datapoint_batch, label_batch, _ in train_dataloader:  # will be tuple of n by 1
        #     train_ex_batch = np.concatenate((datapoint_batch, np.expand_dims(label_batch, axis=1)), axis=1)
        #     visual_labelled_dataset = np.concatenate((visual_labelled_dataset, train_ex_batch), axis=0)  # concat the
        #
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
Goal: do the policy gradient update only after the end of a trajectory
'''
def rl_main(args):

    args.rl_batch_steps = 5
    args.num_episodes = 150
    args.mc_alpha = 0

    args.epsilon = 0.25 # try with full policy. and try with using the full vector to compute a reward. But it really is just a multiple. Unless we specifically penalize assigning 0 counts

    args.oracle_clusters = True

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
            MNIST(args.data_path, return_idx=False),
            batch_size=args.batch_size, drop_last=False
        )

        train_dataset = MNIST(args.data_path)
        print(len(train_dataset))
        args.num_images = 60000
        args.budget = 1  # how many we can label at each round
        args.initial_budget = 1
        args.num_classes = 10

        # args.task_model = MNISTNet()


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
    ROLLING_AVG_LEN = args.rl_batch_steps*2
    prev_reward = torch.ones((ROLLING_AVG_LEN,1))
    prev_reward *=10
    print("prev_reward{}".format(prev_reward))

    STATE_SPACE = args.num_classes
    ACTION_SPACE = args.num_classes
    CLASS_DIST_SPACE = args.num_classes

    pol_class_net = PolicyNet(STATE_SPACE + CLASS_DIST_SPACE, ACTION_SPACE ) # gradient, or hessian in the network..; per class accs as well
    pol_optimizer = optim.Adam(pol_class_net.parameters(), lr=5e-2)


    curr_state = torch.zeros((1,STATE_SPACE + CLASS_DIST_SPACE)) #only feed it in the past state directly

    import copy

    # task_model = model.FCNet(num_classes=args.num_classes)
    # inference_model = task_model
    # inference_model.to(args.device)
    task_model = vgg.vgg16_bn(num_classes=args.num_classes)
    task_model =  MNISTNet()
    # task_model = models.resnet18(pretrained=True, num_classes=args.num_classes)

    accuracies = []
    criterion = torch.nn.CrossEntropyLoss()

    # feel like supporting a desparate cause; might delete later

    entire_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    # ask on SO: multi item getting using pytorch, dataloader

    features, labels, idx = next(iter(entire_loader))
    features = features.numpy()[unlabeled_indices] # we should exactly not be using it as this. Actually, it is OK. we are just saing what is labelled and what is not
    labels = np.expand_dims(labels.numpy()[unlabeled_indices], 1)
    idx =    np.expand_dims(idx.numpy()[unlabeled_indices], 1)
    # X = np.hstack((features,labels ,idx )) #strange that this doesn't work

    # X = np.concatenate((features.reshape(len(features),-1), labels,idx), axis=1)

    args.feature_length = 784
    features = features.reshape(-1,args.feature_length)
    # features = np.repeat(features[:,np.newaxis,:], 3, axis=1)
        # np.broadcast_to(features, shape=(-1,3,args.feature_length))
    X = np.concatenate((features, labels,idx), axis=1)

    from sklearn.cluster import KMeans
    kmeans_obj = KMeans(n_clusters=args.num_classes, random_state=0)  # we can also fit one kmeans at the very start.
    cluster_preds = kmeans_obj.fit_predict(X[:,0:args.feature_length])


    if args.oracle_clusters:
        unlabelled_dataset = np.concatenate((X, labels), axis=1)

    else:
    # we can also just predict (should be fast) again on new datapoints, using the trained classifier. But why not just memorize
        unlabelled_dataset = np.concatenate((X, np.expand_dims(cluster_preds,axis=1)), axis=1)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # try and predict directly in the data space?
    # try and graph where in the dataset it does it as well.
    # in this case, we would again need some fix of the policy gradient.
    # we can no longer just do an easy cross entropy
    # instead, we would be operating more in the regime of .
    # this is a nice analysis on problems of this type!

    # just about rotating and sculpting to your particular area you want
    # ultra few shot learning with fixed dimension, horizon!
    # contributions: to the field of meta/few shot learning using an active learning with reinforcement learning approach
    # keep on layering intersections, until you get the particular area you want.
    # even the approach of doing few shot learning, using active learning is pretty novel IMO




    # we would be trying to essentially do q learning on the choice of datapoint. but make sure you pick in the data space (not action, but continuous choice of the datapoint)
    # the key is really then, trying to do X
    # we could literally do an entire course of lin alg during the break!

    # really, digging into the problems of policy gradient

    # now let's graph the unlabelled dataset
    for cluster in range(args.num_classes):
        # k_means_data  = unlabelled_dataset[unlabelled_dataset[...,-1]==cluster]
        # fig, ax = plt.subplots()

        k_means_data  = unlabelled_dataset[unlabelled_dataset[:,-1]==cluster]

        ax.scatter(k_means_data  [:,0], k_means_data  [:,1])
        ax.scatter(kmeans_obj.cluster_centers_[cluster][0], kmeans_obj.cluster_centers_[cluster][1], s=100)
        fig.savefig(os.path.join(args.out_path, "cluster_{}".format(cluster)))
        # break

    fig.show()

    gradient_accum = torch.zeros((args.rl_batch_steps, 1), requires_grad=False) # accumulate all the losses

    # try making it an empty thing
    gradient_accum = torch.zeros((args.rl_batch_steps), requires_grad=False) # accumulate all the losses

    # loss.backward(0 => doesn't actually execute an update of the weights. we could probably call loss.backward individually

    batched_accs = []


    # try combining it with the state. and also, just try doing an epsilon greedy policy
    import torch.nn.functional as F
    for i in tqdm(range(args.num_episodes)):
        pol_optimizer.zero_grad()

        # here we need a fake label, in order to back prop the loss. And don't backprop immediately, instead, get the gradient,
        # hold it, wait for the reward, and then backprop on that quantity
        action_vector = pol_class_net (curr_state )
        # torch.nn.functional.log_softmax(action_vector)
        # action_dist = torch.zeros((1))
        action_dist = torch.distributions.Categorical(probs=F.softmax(action_vector)) #the diff between Softmax and softmax

        # we probably need logsoftmax here too
        print("action dist{}\n, dist probs{}\n, self f.softmax {}\n, self.log softmax{}\n".format(action_vector,action_dist.probs,
                                                                                              F.softmax(action_vector, dim=1),
                                                                                              F.log_softmax(action_vector,
                                                                                                        dim=1))) #intelligent to take the softmax over the right dimension
        # print() #recall logsoftmax and such

        # if torch.rand() < args.epsilon:
        #     pass
        # else:
        # correct_label1, action1 = get_query(action_dist, unlabeled_dataloader, inference_model, args)
        print(curr_state)

        correct_label, action, unlabelled_dataset, rand = get_query_via_kmeans(action_dist, unlabelled_dataset, args)






        if not rand: #still compute the losses to avoid policy collpase
            # print(rand)
            pred_vector = action_vector.view(1,-1)
            correct_label = correct_label # just a k-size list
            loss = criterion(pred_vector, correct_label)

            print("loss stats")
            print(pred_vector, correct_label)




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

        class_counts, total = dataloader_statistics(train_dataloader, args.num_classes)
        print("class counts {}".format(class_counts))



        #data loader not subscriptable => we should deal with the indices.
        # we could also combine, and get the uncertainties, but WEIGHTED BY CLASS
        # lets just try the dataloader, but it will be challenging when we have the batch size...
        # print(correct_label)
        print("this is the action taken by the sampler")
        print(action)

        acc, curr_state = environment_step(train_dataloader, solver, task_model,args) #might need to write a bit coupled code. This is OK for now
        accuracies.append(acc)

        # curr_state = torch.cat((curr_state_accs, class_counts.t()), axis=1)
        if not rand:

            reward, prev_reward = compute_reward(curr_state, i, prev_reward,args) # basline is around 1% improvement
            print("prev_reward{}".format(prev_reward))
            print("curr reward{}".format(reward))

            # print("log loss is")
            # print(loss)

            # check what the norm of the policy is
            # if torch.sum(loss) <= 0.005:
            #     loss +=0.005 #to avoid policy collapse

            loss *= reward # calling loss backwards here works
            loss *= -1 #want to maximize the reward

            args.penalty_type = "kl"
            p_dist = curr_state[:,args.num_classes:].clone()

            if args.penalty_type == "kl":

            # add the penalty as well
                p_dist /= torch.sum(p_dist) #normalize


            # KL penalty
                q_dist = torch.ones((1, args.num_classes),requires_grad=True)
                q_dist = q_dist* 1/(args.num_classes) #normalize this


            # add delta smoothing
                mcp_loss  = mode_collapse_penalty_kl(action_dist.probs.clone(), q_dist )
            else:

            # Square penalty
                q_dist = torch.ones((1, args.num_classes), requires_grad=True)
                q_dist = q_dist  * i//args.num_classes+1
                mcp_loss = mode_collapse_penalty(p_dist, q_dist)

            print(loss, mcp_loss)

            # loss = mcp_loss #this detracts from the reward
            loss = loss + args.mc_alpha * mcp_loss
            print("total loss")
            print(loss)

            gradient_accum[i% args.rl_batch_steps] = loss

            # tess = torch.mean(gradient_accum)
            # print('tess')
            # print(tess)
            # tess.backward()

        if i % args.rl_batch_steps==0 and i!=0:

            # HER buffer dataloader here: we remember what the choice was, and the reward. then we can decouple the updates!
            # but generally, we should try the baseline (easy)

            print("the gradient is")
            print(gradient_accum)


            # let's prevent the policy collapse
            gradient_accum = gradient_accum[gradient_accum.nonzero()] #filter out the points where we took the epsilon policy

            print(gradient_accum)
            # gradient_accum = torch.clamp(gradient_accum, -10, 10)
            # torch.mean(gradient_accum, dim=0).backward()
            if len(gradient_accum) > 0:
                batched_loss = torch.mean(gradient_accum, dim=0)
                print(batched_loss )
                batched_loss.backward()


                pol_optimizer.step()

            # print(list(pol_class_net.parameters())[0].grad )

            gradient_accum = torch.zeros((args.rl_batch_steps), requires_grad=False)  # accumulate all the losses
            batched_accs.append(acc)

            # now on the next step, you want to run some gradient and see how it goes. and only graph that. Equivalently,
            # just graph every 10th datapoint

            # args.epsilon *= 0.6
             #perform the gradient update
        #     compute the reward. store the gradients
        # store all the gradients, then torch.mean them, and then take a step. This means we only have 10/50 steps.

        # loss.backward()
        # pol_optimizer.step()

        with open(os.path.join(args.out_path, "accs.txt"), "a") as acc_file:
            acc_file.write("{};{}\n".format(acc, curr_state))


        print(curr_state)
        print(acc)

        # with open(os.path.join(args.out_path, "rl_current_accs.txt"), "a") as acc_file:
        #     acc_file.write("{} {}\n".format(acc, class_accs))

        # inference_model = task_model
        # inference_model.to(args.device)
        # task_model = model.FCNet(num_classes=args.num_classes) # remake a new task model each time
        task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        task_model = MNISTNet()
        # task_model = models.resnet18(pretrained=True, num_classes=args.num_classes)

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
        # visual_labelled_dataset = np.zeros((0,3)) #each dimension does not require something new!
        #
        # new_datapoints= np.reshape(np.asarray(action[0]), newshape=(-1,2))
        # for datapoint_batch, label_batch, _ in train_dataloader: #will be tuple of n by 1
        #     train_ex_batch = np.concatenate((datapoint_batch, np.expand_dims(label_batch,axis=1)), axis=1)
        #     visual_labelled_dataset = np.concatenate((visual_labelled_dataset, train_ex_batch), axis=0 ) #concat the

        # visualize_training_dataset(i, args.num_classes, visual_labelled_dataset, new_datapoints)
            #     stack all of them!
            # and furthermore, we need to do a group by on the label.



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
        # args.sampling_method = "random"
        rl_main(args)
        # random_baseline(args)