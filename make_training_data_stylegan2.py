import os
import sys

sys.path.append("..")
import argparse
import copy
import json
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from models.stylegan2 import Generator

from utils.utils import latent_to_image_stylegan2, Interpolate

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_stylegan(args):
    if args["stylegan_ver"] == "2":
        if args["category"] == "car":
            resolution = 512
        elif args["category"] == "face":
            resolution = 1024
        elif args["category"] == "cat":
            resolution = 256
        else:
            assert "Not implementated!"

        with open(args["stylegan_checkpoint"], "rb") as f:
            G_original = pickle.load(f)["G_ema"].float()

        G = Generator(**G_original.init_kwargs)
        G.load_state_dict(G_original.state_dict())
        G.to(device).eval()

    else:
        assert "Not implementated error"

    res = args["dim"][1]
    mode = args["upsample_mode"]
    upsamplers = [
        nn.Upsample(scale_factor=res / 4, mode=mode),
        nn.Upsample(scale_factor=res / 8, mode=mode),
        nn.Upsample(scale_factor=res / 8, mode=mode),
        nn.Upsample(scale_factor=res / 16, mode=mode),
        nn.Upsample(scale_factor=res / 16, mode=mode),
        nn.Upsample(scale_factor=res / 32, mode=mode),
        nn.Upsample(scale_factor=res / 32, mode=mode),
        nn.Upsample(scale_factor=res / 64, mode=mode),
        nn.Upsample(scale_factor=res / 64, mode=mode),
        nn.Upsample(scale_factor=res / 128, mode=mode),
        nn.Upsample(scale_factor=res / 128, mode=mode),
        nn.Upsample(scale_factor=res / 256, mode=mode),
        nn.Upsample(scale_factor=res / 256, mode=mode),
    ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if resolution > 512:
        upsamplers.append(Interpolate(res, "bilinear"))
        upsamplers.append(Interpolate(res, "bilinear"))

    return G, upsamplers


def generate_data(args, num_sample, sv_path, seed):
    if os.path.exists(sv_path):
        pass
    else:
        os.system("mkdir -p %s" % (sv_path))
        print("Experiment folder created at: %s" % (sv_path))

    G, upsamplers = prepare_stylegan(args)

    with torch.no_grad():
        np.random.seed(seed)
        latent_cache = []
        print("Starting to generate samples ...")
        for i in tqdm(range(num_sample)):
            latent = np.random.randn(1, 512)
            latent_cache.append(copy.deepcopy(latent))

            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            img, _= latent_to_image_stylegan2(
                G.mapping, G.synthesis, upsamplers, latent, dim=args["dim"][1], return_upsampled_features=False
            )

            if args["dim"][0] != args["dim"][1]:  # for car dataset case
                img = img[:, 64:448][0]
            else:
                img = img[0]

            img = Image.fromarray(img)
            image_name = os.path.join(sv_path, "image_%d.jpg" % i)
            img.save(image_name)

        latent_cache = np.concatenate(latent_cache, 0)
        latent_sv_path = os.path.join(sv_path, "z_latent_stylegan2.npy")
        np.save(latent_sv_path, latent_cache)

        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--sv_path", type=str)
    args = parser.parse_args()

    opts = json.load(open(args.exp, "r"))
    print("Configurations: ", opts)

    path = opts["exp_dir"]
    if os.path.exists(path):
        pass
    else:
        os.system("mkdir -p %s" % (path))
        print("Experiment folder created at: %s" % (path))

    os.system("cp %s %s" % (args.exp, opts["exp_dir"]))

    generate_data(opts, args.num_sample, args.sv_path, args.seed)
