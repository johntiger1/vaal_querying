import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler

from tqdm import tqdm




class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampling_method = args.sampling_method
        if self.sampling_method == "random":
            self.sampler = sampler.RandomSampler(self.args.budget)
        elif self.sampling_method == "adversary":
            self.sampler = sampler.AdversarySampler(self.args.budget)
        elif self.sampling_method == "uncertainty":
            self.sampler = sampler.UncertaintySampler(self.args.budget)
        elif self.sampling_method == "expected_error":
            self.sampler = sampler.EESampler(self.args.budget)
        elif self.sampling_method == "adversary_1c":
            self.sample = sampler.AdversarySamplerSingleClass(self.args.budget)


        else:
            raise Exception("No valid sampling method provideds")



    def read_data(self, dataloader, labels=True):
        # print(len(dataloader))
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    '''
    
    '''
    def train_without_adv_vae(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader, args):

        labeled_data = self.read_data(querry_dataloader)

        print("the length of the  labelled data is: {}".format(len(querry_dataloader)
                                                               * querry_dataloader.batch_size))
        optim_task_model = optim.Adam(task_model.parameters(), lr=5e-3)

        task_model.train()

        if self.args.cuda:
            task_model = task_model.cuda()
        
        change_lr_iter = self.args.train_iterations // 25

        for iter_count in (range(self.args.train_iterations)):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
    
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9 

            labeled_imgs, labels = next(labeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            # need to expand and de-expand the images
            # labeled_imgs = np.repeat(labeled_imgs[...,np.newaxis,...],3, axis=1)
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)

            # task_loss = task_loss + torch.zeros((1)).log().cuda() #inf also works, interesting!

            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count % 100 == 0:
                print('Current task model loss: {:.4f}'.format(task_loss.item()))


        final_accuracy = self.test(task_model)

        class_based_accs = self.class_based_test(task_model, args.num_classes)
        return final_accuracy, vae, discriminator, class_based_accs

    def train_ret_class_accs(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):

        labeled_data = self.read_data(querry_dataloader)

        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_task_model = optim.Adam(task_model.parameters(), lr=5e-3)

        task_model.train()

        if self.args.cuda:
            task_model = task_model.cuda()

        change_lr_iter = self.args.train_iterations // 25

        for iter_count in tqdm(range(self.args.train_iterations)):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:

                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            if iter_count % 100 == 0:
                print('Current task model loss: {:.4f}'.format(task_loss.item()))

        final_accuracy = self.test(task_model)
        return final_accuracy, vae, discriminator

    '''
    TEST
    '''
    def oracle_train_without_adv_vae(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):

        labeled_data = self.read_data(querry_dataloader)

        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_task_model = optim.Adam(task_model.parameters(), lr=5e-3)

        task_model.train()

        if self.args.cuda:
            task_model = task_model.cuda()

        change_lr_iter = self.args.train_iterations // 25

        for iter_count in tqdm(range(self.args.oracle_train_iterations)):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:

                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            # if iter_count % 100 == 0:
                # print('Current task model loss: {:.4f}'.format(task_loss.item()))

        final_accuracy = self.test(task_model)
        return final_accuracy, vae, discriminator


    def train(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):

        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_task_model = optim.Adam(task_model.parameters(), lr=5e-3)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        task_model.train()

        if self.args.cuda:
            vae = vae.cuda()
            discriminator = discriminator.cuda()
            task_model = task_model.cuda()
        
        change_lr_iter = self.args.train_iterations // 25

        for iter_count in tqdm(range(self.args.train_iterations)):
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
                for param in optim_vae.param_groups:
                    param['lr'] = param['lr'] * 0.9
    
                for param in optim_task_model.param_groups:
                    param['lr'] = param['lr'] * 0.9 

                for param in optim_discriminator.param_groups:
                    param['lr'] = param['lr'] * 0.9 

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()

            # task_model step
            preds = task_model(labeled_imgs)
            task_loss = self.ce_loss(preds, labels)
            optim_task_model.zero_grad()
            task_loss.backward()
            optim_task_model.step()

            # VAE step
            for count in range(self.args.num_vae_steps):
                recon, z, mu, logvar = vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.zeros(labeled_imgs.size(0)).long()
                unlab_real_preds = torch.zeros(unlabeled_imgs.size(0)).long()
                    
                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.ce_loss(labeled_preds, lab_real_preds) + \
                        self.ce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs)
                
                labeled_preds = discriminator(mu)
                # labeled_preds = labeled_out.max(1)[1]
                unlabeled_preds = discriminator(unlab_mu)
                # unlabeled_preds = unlabeled_out.max(1)[1]
                
                lab_real_preds = labels
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).long()

                if self.args.cuda:
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                
                dsc_loss = self.ce_loss(labeled_preds, lab_real_preds) + \
                        self.ce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

                

            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

        final_accuracy = self.test(task_model)
        return final_accuracy, vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, task_learner):
        if self.sampling_method == "random":
            query_indices = self.sampler.sample(unlabeled_dataloader)
        elif self.sampling_method == "uncertainty":
            query_indices = self.sampler.sample(task_learner,
                                                unlabeled_dataloader, 
                                                self.args.cuda)

        elif self.sampling_method == "expected_error":
            query_indices = self.sampler.sample(task_learner,
                                                unlabeled_dataloader,
                                                self.args.cuda)

        elif self.sampling_method == "adversary" or self.sampling_method == "adversary_1c":
            query_indices = self.sampler.sample(vae,
                                                discriminator, 
                                                unlabeled_dataloader, 
                                                self.args.cuda)

        # we can run some analysis on which indices were used in the query
        # a couple of things: class based analysis. also: see how the losses distribute; and other interesting stuff

        # print(query_indices)
        # for x in query_indices:
        #     print(unlabeled_dataloader.dataset[x]) #(X, class label, index)





        query_indices = np.asarray(query_indices).reshape(1,-1)[0,:]
        return query_indices
                

    def class_based_test(self, task_model, num_classes):
        task_model.eval()
        total, correct = torch.zeros((num_classes)), torch.zeros((num_classes))



        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()

            # compute the accuracy per class. Assumes the labels go from 0 to k. This must be passed in

            # good ol' fashioned iterating through the tensor will do
            for y_pred, y in zip(preds, labels):
                # y = y.item()
                total[y.item()] += 1

                if y_pred == y.item():
                    correct[y] += 1


        return correct / total * 100

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
