import torch
import random
import numpy as np

class UncertaintySampler:
    def __init__(self, budget):
        self.budget = budget

    def sample(self, task_learner, data, cuda):
        all_indices = []
        all_preds = []

        for images, _, indices in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                out = task_learner(images)
                preds = out.max(1)[0]
            
            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds *= -1

        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices


class RandomSampler:
    def __init__(self, budget):
        self.budget = budget
    
    def sample(self, data):
        all_indices = []
        for _, _, indices in data:
            all_indices.extend(indices)

        all_indices = [int(x) for x in all_indices]
        random.seed("csc2547")
        new_indices = random.sample(all_indices, self.budget)

        return new_indices

class AdversarySamplerSingleClass:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices

class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                discrim_out = discriminator(mu)
                preds = discrim_out[0] #look at only the probability of the class zero

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        # all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
        
