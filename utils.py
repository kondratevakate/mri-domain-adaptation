from pathlib import Path
import os
import torch

# save_experiment_state
def save_experiment_state(AE, D, AE_opt, D_opt, train_stats, 
                          experiment_name="experiment_1", save_folder="./experiments1/"):
    savepath = os.path.join(save_folder, experiment_name + ".pth")
    directory = os.path.dirname(savepath)
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    state_dict = {
        'train_stats' : train_stats,
        'AE_state_dict': AE.state_dict(),
        'AE_opt_state_dict': AE_opt.state_dict(),
        'D_state_dict': D.state_dict(),
        'D_opt_state_dict': D_opt.state_dict(),
    }
    torch.save(state_dict, savepath)
    print(f"{experiment_name} saved to {savepath}")
    
    # load_experiment_state
def load_experiment_state(AE, D, AE_opt, D_opt, train_stats, 
                          experiment_name="experiment_1", save_folder="./experiments1/"):
    loadpath = os.path.join(save_folder, experiment_name + ".pth")
    state_dict = torch.load(loadpath)
    AE.load_state_dict(state_dict.get("AE_state_dict"))
    AE_opt.load_state_dict(state_dict.get("AE_opt_state_dict"))
    D.load_state_dict(state_dict.get("D_state_dict"))
    D_opt.load_state_dict(state_dict.get("D_opt_state_dict"))
    train_stats.update(state_dict["train_stats"])
    print(f"{experiment_name} loaded.")