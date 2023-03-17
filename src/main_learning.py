import torch
from time import time
from models import meta_model, MetaMorph, shooting_model
from train import train_learning
from prepare_data import load_brats_2021


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    #Parameter of the model
    n_epoch = 50 #Number of epochs
    l = 15 #Number of integration steps
    L2_weight = .5
    lamda = 3e-8 #regularization weight
    v_weight = lamda/l
    z_weight = lamda/l
    inv_weight = 0.001
    mu = 0.01 # weight controlling the intensity addition
    batch_size = 1
    sigma = 6. #smoothing of the gaussian kernel
    debug = True

    #Load data
    train_loader, test_loader, target_img, config = load_brats_2021(device, batch_size)

    #CONFIG dict
    config["debug"] = debug
    config['batch_size'] = batch_size
    config['n_epoch'] = n_epoch
    config["plot_epoch"] = 1
    config["L2_weight"] = L2_weight
    config['v_weight'] = v_weight
    config["z_weight"] = z_weight
    config['inv_weight'] = inv_weight
    config["l"] = l
    config["mu"] = mu
    config["device"] = device
    config["downsample"] = True


    if config["downsample"]:
        ndown = 2
        MNI_img_down = target_img[:, :, ::ndown, ::ndown, ::ndown]
        z0 = torch.zeros(MNI_img_down.shape, dtype=torch.float32)
    else:
        z0 = torch.zeros(target_img.shape, dtype=torch.float32)

    z0.requires_grad = True

    print("### Starting Metamorphoses ###")
    print("L2_weight=", L2_weight)
    print("z_weight=", z_weight)
    print("v_weight=", v_weight)
    print("n_epoch=", n_epoch)
    print("mu=", mu)
    t = time()
    model = MetaMorph(l, z0, device, sigma, mu).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                  weight_decay=1e-8)

    train_learning(model, train_loader, test_loader, target_img, optimizer, config)





