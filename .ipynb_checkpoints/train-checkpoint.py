import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage as nd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, 
                                     ShuffleSplit, LeaveOneGroupOut)
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from IPython.display import clear_output
import matplotlib.pyplot as plt

from viz import *
from utils import *
from models import *

# iteration based
def train_fadernet_iter(train_loader, val_loader, args,
                   AE, D, AE_opt, D_opt, device,
                   AE_crit=nn.MSELoss(), D_crit=nn.BCEWithLogitsLoss(),
                   freq=1, cmap="viridis", 
                   site_labels=[],
                   subgroup_labels=[],
                   train_stats=None,
                   save_freq=0, experiment_name="aenc", save_folder="./experiments1/"):
    
    def plot_results(epoch, freq):
        clear_output(True)
        print("EPOCH {}".format(epoch))
        
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.plot(train_stats["AE_loss"], label="AE_loss")
        plt.plot(train_stats["rec_loss"], label="train_rec_loss")
        plt.plot(train_stats["val rec_loss"], label="val rec_loss")
        plt.subplot(122)
        plt.plot(train_stats["D_loss"], label="D_loss")
        plt.plot([x["mean"] for x in train_stats["SVM_auc_CV"]], label="SVM_auc_CV")
        plt.plot([x["mean"] for x in train_stats["SVM_auc_LSO"]], label="SVM_auc_LSO")
        plt.legend()
        plt.show()
        
        print("Train recon")
        i = np.random.choice(len(train_loader.dataset))
        img, _, attrs = train_loader.dataset[i]
        print(torch.argmax(attrs))
        display_brain_img(img.detach().numpy())
        img, attrs = img.unsqueeze(dim=0).to(device), attrs.unsqueeze(dim=0).to(device)
        img_recon, _ = AE(img, attrs)
        display_brain_img(img_recon.cpu().detach().numpy()[0])
        print("Val recon")
        i = np.random.choice(len(val_loader.dataset))
        img, _, attrs = val_loader.dataset[i]
        print(torch.argmax(attrs))
        display_brain_img(img.detach().numpy())
        img, attrs = img.unsqueeze(dim=0).to(device), attrs.unsqueeze(dim=0).to(device)
        img_recon, _ = AE(img, attrs)
        display_brain_img(img_recon.cpu().detach().numpy()[0])
        
        print("  AE loss: \t\t\t{:.6f}".format(train_stats["AE_loss"][-1]))
        print("  rec loss (in-iteration): \t{:.6f}".format(train_stats["rec_loss"][-1]))
        print("  D loss: \t\t\t{:.6f}".format(train_stats["D_loss"][-1]))
        print("  val rec loss: \t{:.6f}".format(train_stats["val rec_loss"][-1]))
        auc_cv = train_stats["SVM_auc_CV"][-1]
        print("  SVM auc CV: \t{:.3f} ({:.3f})".format(auc_cv["mean"], auc_cv["std"]))
        auc_lso = train_stats["SVM_auc_LSO"][-1]
        print("  SVM auc LSO: \t{:.3f} ({:.3f})".format(auc_lso["mean"], auc_lso["std"]))
        for i in range(len(epoch_train_stats["SVM_auc_LSO"])):
            print(site_labels[i], np.round(epoch_train_stats["SVM_auc_LSO"][i], 3))
        print()
        
        if (epoch + 1) % freq == 0:
            Z = epoch_train_stats["Z_val"]
            sites = epoch_train_stats["sites_val"]
            pca = PCA(n_components=50)
            Z_red = pca.fit_transform(Z)
            tsne = TSNE(n_components=2)
            Z_2d = tsne.fit_transform(Z_red)

            plt.figure(figsize=(10, 6))
            colors = get_colors(plt.get_cmap(cmap), np.linspace(0.05, 0.9, len(site_labels)))

            if len(subgroup_labels) > 0:
                for sg in np.unique(subgroup_labels):
                    idx_sg = (subgroup_labels == sg)
                    for i in range(len(site_labels)):
                        plt.scatter(Z_2d[(sites == i) * idx_sg, 0], Z_2d[(sites == i) * idx_sg, 1], 
                                    s=30, c=colors[i], marker="${}$".format(sg),
                                    alpha=0.8, label=site_labels[i] if sg == 0 else None)
            else:
                for i in range(len(site_labels)):
                    plt.scatter(Z_2d[sites == i, 0], Z_2d[sites == i, 1], 
                                s=30, c=colors[i], alpha=1, label=site_labels[i])
            plt.legend()
    #         plt.xticks([],[]), plt.yticks([],[])
            plt.show()
        
    train_stats = {
        "D_loss" : [],
        "rec_loss" : [],
        "AE_loss" : [],
        "val rec_loss" : [],
        "SVM_auc_CV" : [],
        "SVM_auc_LSO" : [],
    } if train_stats is None else train_stats
    
    for epoch in range(args["n_epochs"]):
        print("TRAIN EPOCH {}...".format(epoch))
        epoch_train_stats = {
            "D_loss" : [],
            "rec_loss" : [],
            "AE_loss" : [],
            "val rec_loss" : [],
            "Z_val" : [],
            "y_val" : [],
            "sites_val" : [],
            "SVM_auc_CV" : [],
            "SVM_auc_LSO" : [],
        }
        
        D_n_upd = args["D_n_upd"] if args["n_pretr_epochs"] <= epoch < args["n_pretr_epochs"] + args["D_loop_epochs"] else 1
        
        if args["lambda_scheduler"] == "linear":
            if epoch <= args["n_pretr_epochs"]:
                cur_lambda = args["min_lambda"]
            elif cur_lambda < args["max_lambda"]:
                cur_lambda += args["inc_lambda"]
            else: 
                cur_lambda = args["max_lambda"]
        elif args["lambda_scheduler"] == "exp":
            p = epoch / args["n_epochs"]
            cur_lambda = (2 / (1 + np.exp(-args["lambda_gamma"] * p)) - 1) * args["max_lambda"]
        print(cur_lambda)

        D.train(True)
        AE.train(True)
        # here train_loader returns x, y, a 
        for it in tqdm(range(args["iter_per_epoch"])):
            inputs_batch, _, attrs_batch = next(iter(train_loader))
            inputs_batch, attrs_batch = inputs_batch.to(device), attrs_batch.to(device)
            
            # train D
            # use D to predict attrs from latent reprs
            # encode inputs & attrs with AE
            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            for _ in range(D_n_upd):
                outputs_batch = D(latents_batch.detach())
                D_loss = D_crit(outputs_batch, attrs_batch)
                if epoch > args["n_pretr_epochs"]:
                    D_opt.zero_grad()
                    D_loss.backward()
                    D_opt.step()
            epoch_train_stats["D_loss"].append(D_loss.item())
            
            # train AE
            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            rec_loss = AE_crit(recons_batch, inputs_batch)
            outputs_batch = D(latents_batch)
            D_loss = D_crit(outputs_batch, 1 - attrs_batch)   # D_loss = D_crit(outputs_batch, (1 - attrs_batch))
            AE_loss = rec_loss + cur_lambda * D_loss      # AE_loss = rec_loss + cur_lambda * D_loss
            AE_opt.zero_grad()
            AE_loss.backward()
            AE_opt.step()
            epoch_train_stats["rec_loss"].append(rec_loss.item())
            epoch_train_stats["AE_loss"].append(AE_loss.item())
        
        AE.train(False)
        for inputs_batch, targets_batch, attrs_batch in tqdm(val_loader):
            inputs_batch, attrs_batch = inputs_batch.to(device), attrs_batch.to(device)

            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            rec_loss = AE_crit(recons_batch, inputs_batch)
            epoch_train_stats["val rec_loss"].append(rec_loss.item())
            epoch_train_stats["Z_val"] += list(latents_batch.cpu().detach().numpy().reshape(len(latents_batch), -1))
            epoch_train_stats["y_val"] += list(targets_batch.cpu().detach().numpy())
            epoch_train_stats["sites_val"] += list(attrs_batch.argmax(dim=1).cpu().detach().numpy())
            
        epoch_train_stats["Z_val"] = np.array(epoch_train_stats["Z_val"])
        epoch_train_stats["y_val"] = np.array(epoch_train_stats["y_val"])
        epoch_train_stats["sites_val"] = np.array(epoch_train_stats["sites_val"])
        
        clf = SVC(kernel="rbf") # linear kernel? 
        # CV
        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        epoch_train_stats["SVM_auc_CV"] = cross_val_score(clf, epoch_train_stats["Z_val"], epoch_train_stats["y_val"], 
                                                          scoring="roc_auc", cv=cv)
        # LSO
        cv = LeaveOneGroupOut()
        epoch_train_stats["SVM_auc_LSO"] = cross_val_score(clf, epoch_train_stats["Z_val"], epoch_train_stats["y_val"], 
                                                          groups=epoch_train_stats["sites_val"], scoring="roc_auc", cv=cv)
        
        
        # add mean stats
        for stat_name in epoch_train_stats:
            if stat_name in train_stats:
                if "auc" in stat_name:
                    train_stats[stat_name].append({"mean" : np.mean(epoch_train_stats[stat_name]),
                                                   "std" : np.std(epoch_train_stats[stat_name])})
                else:
                    train_stats[stat_name].append(np.mean(epoch_train_stats[stat_name]))
        
        plot_results(epoch, freq)

        if (save_freq > 0) and ((epoch + 1) % freq == 0):
            save_experiment_state(AE, D, AE_opt, D_opt, train_stats, experiment_name, save_folder)
        
    return train_stats


# sample-based
def train_fadernet_sample(train_loader, val_loader, args,
                   AE, D, AE_opt, D_opt, device,
                   AE_crit=nn.MSELoss(), D_crit=nn.BCEWithLogitsLoss(),
                   freq=1, cmap="viridis", 
                   site_labels=[],
                   subgroup_labels=[],
                   train_stats=None,
                   save_freq=0, experiment_name="aenc", save_folder="./experiments1/"):
    
    def plot_results(epoch, freq):
        clear_output(True)
        print("EPOCH {}".format(epoch))
        
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.plot(train_stats["AE_loss"], label="AE_loss")
        plt.plot(train_stats["rec_loss"], label="train_rec_loss")
        plt.plot(train_stats["val rec_loss"], label="val rec_loss")
        plt.subplot(122)
        plt.plot(train_stats["D_loss"], label="D_loss")
        plt.plot([x["mean"] for x in train_stats["SVM_auc_CV"]], label="SVM_auc_CV")
        plt.plot([x["mean"] for x in train_stats["SVM_auc_LSO"]], label="SVM_auc_LSO")
        plt.legend()
        plt.show()
        
        print("Train recon")
        i = np.random.choice(len(train_loader.dataset))
        img, _, attrs = train_loader.dataset[i]
        print(torch.argmax(attrs))
        display_brain_img(img.detach().numpy())
        img, attrs = img.unsqueeze(dim=0).to(device), attrs.unsqueeze(dim=0).to(device)
        img_recon, _ = AE(img, attrs)
        display_brain_img(img_recon.cpu().detach().numpy()[0])
        print("Val recon")
        i = np.random.choice(len(val_loader.dataset))
        img, _, attrs = val_loader.dataset[i]
        print(torch.argmax(attrs))
        display_brain_img(img.detach().numpy())
        img, attrs = img.unsqueeze(dim=0).to(device), attrs.unsqueeze(dim=0).to(device)
        img_recon, _ = AE(img, attrs)
        display_brain_img(img_recon.cpu().detach().numpy()[0])
        
        print("  AE loss: \t\t\t{:.6f}".format(train_stats["AE_loss"][-1]))
        print("  rec loss (in-iteration): \t{:.6f}".format(train_stats["rec_loss"][-1]))
        print("  D loss: \t\t\t{:.6f}".format(train_stats["D_loss"][-1]))
        print("  val rec loss: \t{:.6f}".format(train_stats["val rec_loss"][-1]))
        auc_cv = train_stats["SVM_auc_CV"][-1]
        print("  SVM auc CV: \t{:.3f} ({:.3f})".format(auc_cv["mean"], auc_cv["std"]))
        auc_lso = train_stats["SVM_auc_LSO"][-1]
        print("  SVM auc LSO: \t{:.3f} ({:.3f})".format(auc_lso["mean"], auc_lso["std"]))
        for i in range(len(epoch_train_stats["SVM_auc_LSO"])):
            print(site_labels[i], np.round(epoch_train_stats["SVM_auc_LSO"][i], 3))
        print()
        
        if (epoch + 1) % freq == 0:
            Z = epoch_train_stats["Z_val"]
            sites = epoch_train_stats["sites_val"]
            pca = PCA(n_components=50)
            Z_red = pca.fit_transform(Z)
            tsne = TSNE(n_components=2)
            Z_2d = tsne.fit_transform(Z_red)

            plt.figure(figsize=(10, 6))
            colors = get_colors(plt.get_cmap(cmap), np.linspace(0.05, 0.9, len(site_labels)))

            if len(subgroup_labels) > 0:
                for sg in np.unique(subgroup_labels):
                    idx_sg = (subgroup_labels == sg)
                    for i in range(len(site_labels)):
                        plt.scatter(Z_2d[(sites == i) * idx_sg, 0], Z_2d[(sites == i) * idx_sg, 1], 
                                    s=30, c=colors[i], marker="${}$".format(sg),
                                    alpha=0.8, label=site_labels[i] if sg == 0 else None)
            else:
                for i in range(len(site_labels)):
                    plt.scatter(Z_2d[sites == i, 0], Z_2d[sites == i, 1], 
                                s=30, c=colors[i], alpha=1, label=site_labels[i])
            plt.legend()
    #         plt.xticks([],[]), plt.yticks([],[])
            plt.show()
        
    train_stats = {
        "D_loss" : [],
        "rec_loss" : [],
        "AE_loss" : [],
        "val rec_loss" : [],
        "SVM_auc_CV" : [],
        "SVM_auc_LSO" : [],
    } if train_stats is None else train_stats
    
    for epoch in range(args["n_epochs"]):
        print("TRAIN EPOCH {}...".format(epoch))
        epoch_train_stats = {
            "D_loss" : [],
            "rec_loss" : [],
            "AE_loss" : [],
            "val rec_loss" : [],
            "Z_val" : [],
            "y_val" : [],
            "sites_val" : [],
            "SVM_auc_CV" : [],
            "SVM_auc_LSO" : [],
        }
        
        D_n_upd = args["D_n_upd"] if args["n_pretr_epochs"] <= epoch < args["n_pretr_epochs"] + args["D_loop_epochs"] else 1
        
        if args["lambda_scheduler"] == "linear":
            if epoch <= args["n_pretr_epochs"]:
                cur_lambda = args["min_lambda"]
            elif cur_lambda < args["max_lambda"]:
                cur_lambda += args["inc_lambda"]
            else: 
                cur_lambda = args["max_lambda"]
        elif args["lambda_scheduler"] == "exp":
            p = epoch / args["n_epochs"]
            cur_lambda = (2 / (1 + np.exp(-args["lambda_gamma"] * p)) - 1) * args["max_lambda"]
        print(cur_lambda)
        
        D.train(True)
        AE.train(True)
        # here train_loader returns x, y, a 
        for inputs_batch, _, attrs_batch in tqdm(train_loader):
            inputs_batch, attrs_batch = inputs_batch.to(device), attrs_batch.to(device)
            
            # train D
            # use D to predict attrs from latent reprs
            # encode inputs & attrs with AE
            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            for _ in range(D_n_upd):
                outputs_batch = D(latents_batch.detach())
                D_loss = D_crit(outputs_batch, attrs_batch)
                if epoch > args["n_pretr_epochs"]:
                    D_opt.zero_grad()
                    D_loss.backward()
                    D_opt.step()
            epoch_train_stats["D_loss"].append(D_loss.item())
            
            # train AE
            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            rec_loss = AE_crit(recons_batch, inputs_batch)
            outputs_batch = D(latents_batch)
            D_loss = D_crit(outputs_batch, 1 - attrs_batch)   # D_loss = D_crit(outputs_batch, (1 - attrs_batch))
            AE_loss = rec_loss + cur_lambda * D_loss      # AE_loss = rec_loss + cur_lambda * D_loss
            AE_opt.zero_grad()
            AE_loss.backward()
            AE_opt.step()
            epoch_train_stats["rec_loss"].append(rec_loss.item())
            epoch_train_stats["AE_loss"].append(AE_loss.item())
            
        
        AE.train(False)
        for inputs_batch, targets_batch, attrs_batch in tqdm(val_loader):
            inputs_batch, attrs_batch = inputs_batch.to(device), attrs_batch.to(device)

            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            rec_loss = AE_crit(recons_batch, inputs_batch)
            epoch_train_stats["val rec_loss"].append(rec_loss.item())
            epoch_train_stats["Z_val"] += list(latents_batch.cpu().detach().numpy().reshape(len(latents_batch), -1))
            epoch_train_stats["y_val"] += list(targets_batch.cpu().detach().numpy())
            epoch_train_stats["sites_val"] += list(attrs_batch.argmax(dim=1).cpu().detach().numpy())
            
        epoch_train_stats["Z_val"] = np.array(epoch_train_stats["Z_val"])
        epoch_train_stats["y_val"] = np.array(epoch_train_stats["y_val"])
        epoch_train_stats["sites_val"] = np.array(epoch_train_stats["sites_val"])
        
        clf = SVC(kernel="rbf") # linear kernel? 
        # CV
        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        epoch_train_stats["SVM_auc_CV"] = cross_val_score(clf, epoch_train_stats["Z_val"], epoch_train_stats["y_val"], 
                                                          scoring="roc_auc", cv=cv)
        # LSO
        cv = LeaveOneGroupOut()
        epoch_train_stats["SVM_auc_LSO"] = cross_val_score(clf, epoch_train_stats["Z_val"], epoch_train_stats["y_val"], 
                                                          groups=epoch_train_stats["sites_val"], scoring="roc_auc", cv=cv)
        
        
        # add mean stats
        for stat_name in epoch_train_stats:
            if stat_name in train_stats:
                if "auc" in stat_name:
                    train_stats[stat_name].append({"mean" : np.mean(epoch_train_stats[stat_name]),
                                                   "std" : np.std(epoch_train_stats[stat_name])})
                else:
                    train_stats[stat_name].append(np.mean(epoch_train_stats[stat_name]))
        
        plot_results(epoch, freq)

        if (save_freq > 0) and ((epoch + 1) % freq == 0):
            save_experiment_state(AE, D, AE_opt, D_opt, train_stats, experiment_name, save_folder)
        
    return train_stats

def train_fadernet_sched(
    train_loader, val_loader, args,
    AE, D, AE_opt, D_opt, device,
    AE_crit = nn.MSELoss(), D_crit = nn.BCEWithLogitsLoss(),
    score_freq = 1, vis_freq = 1, cmap = "viridis", 
    site_labels = [],
    subgroup_labels = [],
    train_stats = None,
    save_freq = 0, 
    experiment_name = "aenc",
    save_folder = "", 
    experiment = False):
    
    def plot_results(epoch, score_freq, vis_freq, experiment, last_epoch=False):
        clear_output(True)
        print("EPOCH {}".format(epoch))
        
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.plot(train_stats["AE_loss"], label="AE_loss")
        plt.plot(train_stats["rec_loss"], label="train_rec_loss")
        plt.plot(train_stats["val_rec_loss"], label="val_rec_loss")
        plt.subplot(122)
        plt.plot(train_stats["D_loss"][score_freq - 1::score_freq], label="D_loss") #
        if len(train_stats["SVM_auc_CV"]) > 0:
            plt.plot([x["mean"] for x in train_stats["SVM_auc_CV"]], label="SVM_auc_CV")
            plt.plot([x["mean"] for x in train_stats["SVM_auc_LSO"]], label="SVM_auc_LSO")
        plt.legend()
        plt.show()
        
        print("Train recon")
        i = np.random.choice(len(train_loader.dataset))
        img, _, attrs = train_loader.dataset[i]
        print(torch.argmax(attrs))
        display_brain_img(img.detach().numpy())
        
        img, attrs = img.unsqueeze(dim=0).to(device), attrs.unsqueeze(dim=0).to(device)
        img_recon, _ = AE(img, attrs)
        display_brain_img(img_recon.cpu().detach().numpy()[0])
        print("Val recon")
        
        i = np.random.choice(len(val_loader.dataset))
        img, _, attrs = val_loader.dataset[i]
        print(torch.argmax(attrs))
        display_brain_img(img.detach().numpy())
        img, attrs = img.unsqueeze(dim=0).to(device), attrs.unsqueeze(dim=0).to(device)
        img_recon, _ = AE(img, attrs)
        display_brain_img(img_recon.cpu().detach().numpy()[0])
        
        print("  AE loss: \t\t\t{:.6f}".format(train_stats["AE_loss"][-1]))
        print("  rec loss (in-iteration): \t{:.6f}".format(train_stats["rec_loss"][-1]))
        print("  D loss: \t\t\t{:.6f}".format(train_stats["D_loss"][-1]))
        print("  val rec loss: \t{:.6f}".format(train_stats["val_rec_loss"][-1]))
        if len(train_stats["SVM_auc_CV"]) > 0:
            auc_cv = train_stats["SVM_auc_CV"][-1]
            print("  SVM auc CV: \t{:.3f} ({:.3f})".format(auc_cv["mean"], auc_cv["std"]))
            auc_lso = train_stats["SVM_auc_LSO"][-1]
            print("  SVM auc LSO: \t{:.3f} ({:.3f})".format(auc_lso["mean"], auc_lso["std"]))
            for i in range(len(epoch_train_stats["SVM_auc_LSO"])):
                print(site_labels[i], np.round(epoch_train_stats["SVM_auc_LSO"][i], 3))
        print()
        
        if last_epoch or ((epoch + 1) % vis_freq == 0):
            display_brain_img(torch.squeeze(img, dim = 0).cpu().detach().numpy(), experiment = experiment, 
                              figure_name = 'Validation original')
            display_brain_img(img_recon.cpu().detach().numpy()[0], 
                          experiment = experiment, 
                          figure_name = 'Validation reconstruction')
            Z = epoch_train_stats["Z_val"]
            sites = epoch_train_stats["sites_val"]
            pca = PCA(n_components=50)
            Z_red = pca.fit_transform(Z)
            tsne = TSNE(n_components=2)
            Z_2d = tsne.fit_transform(Z_red)

            plt.figure(figsize=(10, 6))
            colors = get_colors(plt.get_cmap(cmap), np.linspace(0.05, 0.9, len(site_labels)))

            if len(subgroup_labels) > 0:
                for sg in np.unique(subgroup_labels):
                    idx_sg = (subgroup_labels == sg)
                    for i in range(len(site_labels)):
                        plt.scatter(Z_2d[(sites == i) * idx_sg, 0], Z_2d[(sites == i) * idx_sg, 1], 
                                    s=30, c=colors[i], marker="${}$".format(sg),
                                    alpha=0.8, label=site_labels[i] if sg == 0 else None)
            else:
                for i in range(len(site_labels)):
                    plt.scatter(Z_2d[sites == i, 0], Z_2d[sites == i, 1], 
                                s=30, c=colors[i], alpha=1, label=site_labels[i])
            plt.legend()
            
            if experiment:
                experiment.log_figure(figure_name = 'Latent vectors TSNE Embedding')
            else:
                plt.show()
        
    train_stats = {
        "D_loss" : [],
        "rec_loss" : [],
        "AE_loss" : [],
        "val_rec_loss" : [],
        "SVM_auc_CV" : [],
        "SVM_auc_LSO" : [],
    } if train_stats is None else train_stats
    
    # epoch = len(train_stats["val rec_loss"])
    for epoch in range(args["n_epochs"]):
        print("TRAIN EPOCH {}...".format(epoch))
        epoch_train_stats = {
            "D_loss" : [],
            "rec_loss" : [],
            "AE_loss" : [],
            "val_rec_loss" : [],
            "Z_val" : [],
            "y_val" : [],
            "sites_val" : [],
            "SVM_auc_CV" : [],
            "SVM_auc_LSO" : [],
        }
        last_epoch = False

        cur_val_loss = train_stats["val_rec_loss"][-1] if train_stats["val_rec_loss"] else np.inf
        D_n_upd = args["D_n_upd"] #if args["n_pretr_epochs"] <= cur_loss < args["n_pretr_epochs"] + args["D_loop_epochs"] else 1
        
        if cur_val_loss >= args["pretr_val_loss"]:
            cur_lambda = args["min_lambda"]
        elif cur_val_loss > args["min_val_loss"]:
            cur_lambda = args["min_lambda"] + (args["max_lambda"] - args["min_lambda"]) * (
                (args["pretr_val_loss"] - cur_val_loss) / (args["pretr_val_loss"] - args["min_val_loss"]))
        else: 
            cur_lambda = args["max_lambda"]
        print(cur_lambda)
        
        D.train(True)
        AE.train(True)
        # here train_loader returns x, y, a 
        for inputs_batch, _, attrs_batch in tqdm(train_loader):
            inputs_batch, attrs_batch = inputs_batch.to(device), attrs_batch.to(device)
            
            # train D
            # use D to predict attrs from latent reprs
            # encode inputs & attrs with AE
            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            for _ in range(D_n_upd):
                outputs_batch = D(latents_batch.detach())
                D_loss = D_crit(outputs_batch, attrs_batch)
                # if cur_val_loss < args["pretr_val_loss"]:
                if cur_lambda > 0:
                    D_opt.zero_grad()
                    D_loss.backward()
                    D_opt.step()
            epoch_train_stats["D_loss"].append(D_loss.item())
            if experiment:
                experiment.log_metric("D_loss", D_loss.item())
            
            # train AE
            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            rec_loss = AE_crit(recons_batch, inputs_batch)
            outputs_batch = D(latents_batch)
            D_loss = D_crit(outputs_batch, 1 - attrs_batch)   # D_loss = D_crit(outputs_batch, (1 - attrs_batch))
            AE_loss = rec_loss + cur_lambda * D_loss      # AE_loss = rec_loss + cur_lambda * D_loss
            AE_opt.zero_grad()
            AE_loss.backward()
            AE_opt.step()
            epoch_train_stats["rec_loss"].append(rec_loss.item())
            epoch_train_stats["AE_loss"].append(AE_loss.item())
            if experiment:
                experiment.log_metric("AE_loss", AE_loss.item())
                experiment.log_metric("rec_loss", rec_loss.item())
            
        
        AE.train(False)
        # cross-validation of the inforativeness of latent vectors
        for inputs_batch, targets_batch, attrs_batch in tqdm(val_loader):
            inputs_batch, attrs_batch = inputs_batch.to(device), attrs_batch.to(device)

            recons_batch, latents_batch = AE(inputs_batch, attrs_batch)
            rec_loss = AE_crit(recons_batch, inputs_batch)
            epoch_train_stats["val_rec_loss"].append(rec_loss.item())
            epoch_train_stats["Z_val"] += list(latents_batch.cpu().detach().numpy().reshape(len(latents_batch), -1))
            epoch_train_stats["y_val"] += list(targets_batch.cpu().detach().numpy())
            epoch_train_stats["sites_val"] += list(attrs_batch.argmax(dim=1).cpu().detach().numpy())
            if experiment:
                experiment.log_metric("val_rec_loss", rec_loss.item())
            
        epoch_train_stats["Z_val"] = np.array(epoch_train_stats["Z_val"])
        epoch_train_stats["y_val"] = np.array(epoch_train_stats["y_val"])
        epoch_train_stats["sites_val"] = np.array(epoch_train_stats["sites_val"])

        # here the training-validation phase is finished
        # we check if it was the last epoch
        # perform scoring and visualization + save model if 
        # - it is last_epoch
        # - it is score_freq / vis_freq / save_freq
        # after visualization is done - break if it was the last_epoch
        if np.mean(epoch_train_stats["val_rec_loss"]) < args["stop_val_loss"]:
            last_epoch = True
        
        if last_epoch or ((epoch + 1) % score_freq == 0):
            clf = SVC(kernel="rbf") # linear kernel? 
            # CV
            cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
            epoch_train_stats["SVM_auc_CV"] = cross_val_score(clf, epoch_train_stats["Z_val"], epoch_train_stats["y_val"], 
                                                              scoring="roc_auc", cv=cv)
            # LSO
            cv = LeaveOneGroupOut()
            epoch_train_stats["SVM_auc_LSO"] = cross_val_score(clf, epoch_train_stats["Z_val"], epoch_train_stats["y_val"], 
                                                              groups=epoch_train_stats["sites_val"], scoring="roc_auc", cv=cv)
            if experiment:
                experiment.log_metric("SVM_auc_LSO_mean", epoch_train_stats["SVM_auc_LSO"].mean())
                experiment.log_metric("SVM_auc_LSO_std", epoch_train_stats["SVM_auc_LSO"].std())
                experiment.log_metric("SVM_auc_CV_mean", epoch_train_stats["SVM_auc_CV"].mean())
                experiment.log_metric("SVM_auc_CV_std", epoch_train_stats["SVM_auc_CV"].std())
                
        # add mean stats
        for stat_name in epoch_train_stats:
            if stat_name in train_stats:
                if ("auc" in stat_name):
                    if len(epoch_train_stats[stat_name]) > 0:
                        train_stats[stat_name].append({"mean" : np.mean(epoch_train_stats[stat_name]),
                                                      "std" : np.std(epoch_train_stats[stat_name])})
                else:
                    train_stats[stat_name].append(np.mean(epoch_train_stats[stat_name]))
        

        plot_results(epoch, score_freq, vis_freq, experiment, last_epoch)

        if (save_freq > 0) and (last_epoch or ((epoch + 1) % vis_freq == 0)):
            save_experiment_state(AE, D, AE_opt, D_opt, train_stats, experiment_name, save_folder)
            
        if experiment:
            experiment.log_epoch_end(epoch)

        if last_epoch:
            print("Training is finished.")
            break
        
    return train_stats