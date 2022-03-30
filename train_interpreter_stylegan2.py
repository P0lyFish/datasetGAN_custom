import os
import argparse
import sys

sys.path.append("..")

import torch
import torch.nn as nn
import imageio
import json
import numpy as np
from PIL import Image
import gc

import pickle
from models.stylegan2 import Generator
from resnet import Resnet
from torch.distributions import Categorical
import scipy.stats
from utils.utils import multi_acc, colorize_mask, get_label_stas, oht_to_scalar, Interpolate, latent_to_image_stylegan2, get_optimizer, get_scheduler, get_real_deg
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import warnings
from discriminator import get_all_D

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning)


class trainData(Dataset):
    def __init__(self, q_dim):
        self.q_dim = q_dim

    def __getitem__(self, index):
        latent = np.random.randn(512)
        latent = torch.from_numpy(latent).type(torch.FloatTensor).cuda()

        q = np.random.randn(self.q_dim)

        return latent, q

    def __len__(self):
        return 1000


def latent_to_feat(latent, G, upsamplers, args):
    img, fmap = latent_to_image_stylegan2(
        G, upsamplers, latent, dim=args["dim"][1], return_upsampled_features=True
    )


def prepare_stylegan(args):
    with open(args["stylegan_checkpoint"], "rb") as f:
        G_original = pickle.load(f)["G_ema"].float()

    G = Generator(**G_original.init_kwargs)
    G.load_state_dict(G_original.state_dict())
    G.cuda().eval()

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

    if G.img_resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if G.img_resolution > 512:
        upsamplers.append(Interpolate(res, "bilinear"))
        upsamplers.append(Interpolate(res, "bilinear"))

    return G, upsamplers


def generate_data(args, checkpoint_path, num_sample, start_step=0, vis=True):
    if args["category"] == "face":
        from utils.data_util import face_palette as palette
    else:
        raise NotImplementedError

    if not vis:
        result_path = os.path.join(checkpoint_path, "samples")
    else:
        result_path = os.path.join(checkpoint_path, "vis_%d" % num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system("mkdir -p %s" % (result_path))
        print("Experiment folder created at: %s" % (result_path))

    G, upsamplers = prepare_stylegan(args)

    for MODEL_NUMBER in range(args["model_num"]):
        classifier = pixel_classifier(numpy_class=args["number_class"], dim=args["dim"][-1])
        classifier.to(device)
        checkpoint = torch.load(os.path.join(checkpoint_path, "model_" + str(MODEL_NUMBER) + ".pth"))
        classifier.load_state_dict(checkpoint["model_state_dict"])

        classifier.eval()
        classifier_list.append(classifier)

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = start_step

        print("num_sample: ", num_sample)

        for i in tqdm(range(num_sample)):
            if i % 100 == 0:
                print("Genearte", i, "Out of:", num_sample)

            curr_result = {}

            latent = np.random.randn(1, 512)

            curr_result["latent"] = latent

            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)

            img, affine_layers = latent_to_image_stylegan2(
                G, upsamplers, latent, dim=args["dim"][1], return_upsampled_features=True
            )

            if args["dim"][0] != args["dim"][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]

            image_cache.append(img)
            if args["dim"][0] != args["dim"][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]

            affine_layers = affine_layers.reshape(args["dim"][-1], -1).transpose(1, 0)

            all_seg = []
            all_entropy = []
            mean_seg = None

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args["model_num"]):
                classifier = classifier_list[MODEL_NUMBER]

                img_seg = classifier(affine_layers)

                img_seg = img_seg.squeeze()

                entropy = Categorical(logits=img_seg).entropy()
                all_entropy.append(entropy)

                all_seg.append(img_seg)
                if mean_seg is None:
                    mean_seg = softmax_f(img_seg)
                else:
                    mean_seg += softmax_f(img_seg)

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args["dim"][0], args["dim"][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()

                seg_mode_ensemble.append(img_seg_final)

            mean_seg = mean_seg / len(all_seg)

            full_entropy = Categorical(mean_seg).entropy()

            js = full_entropy - torch.mean(torch.stack(all_entropy), 0)

            top_k = js.sort()[0][-int(js.shape[0] / 10) :].mean()
            entropy_calculate.append(top_k)

            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args["dim"][0], args["dim"][1])

            del affine_layers

            if vis:
                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + ".jpg"), color_mask.astype(np.uint8))
                imageio.imwrite(os.path.join(result_path, "vis_" + str(i) + "_image.jpg"), img.astype(np.uint8))
            else:
                seg_cache.append(img_seg_final)
                curr_result["uncertrainty_score"] = top_k.item()

                image_name = os.path.join(result_path, str(count_step) + ".png")
                img = Image.fromarray(img)
                img.save(image_name)

                image_label_name = os.path.join(result_path, "label_" + str(count_step) + ".npy")
                img_seg = img_seg_final.astype("uint8")
                np.save(image_label_name, img_seg)

                js_name = os.path.join(result_path, str(count_step) + ".npy")
                js = js.cpu().numpy().reshape(args["dim"][0], args["dim"][1])
                np.save(js_name, js)

                curr_result["image_name"] = image_name
                curr_result["image_label_name"] = image_label_name
                curr_result["js_name"] = js_name
                count_step += 1

                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + ".pickle"), "wb") as f:
                        pickle.dump(results, f)

        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + ".pickle"), "wb") as f:
            pickle.dump(results, f)


def prepare_data(args, palette):
    G, upsamplers = prepare_stylegan(args)
    latent_all = np.load(args["annotation_image_latent_path"])
    latent_all = torch.from_numpy(latent_all).cuda()

    # Load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[: args["max_training"]]
    num_data = len(latent_all)

    for i in tqdm(range(len(latent_all))):
        if i >= args["max_training"]:
            break
        name = "image_%0d.npy" % i

        im_frame = np.load(os.path.join(args["annotation_mask_path"], name))
        mask = np.array(im_frame)
        mask = cv2.resize(np.squeeze(mask), dsize=(args["dim"][1], args["dim"][0]), interpolation=cv2.INTER_NEAREST)

        mask_list.append(mask)

        im_name = os.path.join(args["annotation_mask_path"], "image_%d.jpg" % i)
        img = Image.open(im_name)
        img = img.resize((args["dim"][1], args["dim"][0]))

        im_list.append(np.array(img))

    # Delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0

    all_mask = np.stack(mask_list)

    # 3. Generate ALL training data for training pixel classifier
    all_feature_maps_train = np.zeros(
        (args["dim"][0] * args["dim"][1] * len(latent_all), args["dim"][2]), dtype=np.float16
    )
    all_mask_train = np.zeros((args["dim"][0] * args["dim"][1] * len(latent_all),), dtype=np.float16)

    vis = []
    for i in tqdm(range(len(latent_all))):
        gc.collect()
        latent_input = latent_all[i].float()

        img, feature_maps = latent_to_image_stylegan2(
            G,
            upsamplers,
            latent_input.unsqueeze(0),
            dim=args["dim"][1],
            return_upsampled_features=True,
            use_style_latents=args["annotation_data_from_w"],
        )

        if args["dim"][0] != args["dim"][1]:  # for car
            img = img[:, 64:448]
            feature_maps = feature_maps[:, :, 64:448]

        mask = all_mask[i : i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)
        feature_maps = feature_maps.reshape(-1, args["dim"][2])
        new_mask = np.squeeze(mask)

        mask = mask.reshape(-1)
        all_feature_maps_train[
            args["dim"][0] * args["dim"][1] * i : args["dim"][0] * args["dim"][1] * i + args["dim"][0] * args["dim"][1]
        ] = (feature_maps.cpu().detach().numpy().astype(np.float16))
        all_mask_train[
            args["dim"][0] * args["dim"][1] * i : args["dim"][0] * args["dim"][1] * i + args["dim"][0] * args["dim"][1]
        ] = mask.astype(np.float16)

        img_show = cv2.resize(
            np.squeeze(img[0]), dsize=(args["dim"][1], args["dim"][1]), interpolation=cv2.INTER_NEAREST
        )
        curr_vis = np.concatenate([im_list[i], img_show, colorize_mask(new_mask, palette)], 0)

        vis.append(curr_vis)

    vis = np.concatenate(vis, 1)
    imageio.imwrite(os.path.join(args["exp_dir"], "train_data.jpg"), vis)

    return all_feature_maps_train, all_mask_train, num_data


def main(args):
    if args["category"] == "face":
        from utils.data_util import face_palette as palette
    else:
        raise NotImplementedError

    G_stylegan, upsamplers = prepare_stylegan(args)
    G_mapping = nn.DataParallel(G_stylegan.mapping)
    G_synthesis = nn.DataParallel(G_stylegan.synthesis)
    upsamplers = [nn.DataParallel(upsampler) for upsampler in upsamplers]

    train_data = trainData(args["q_dim"])

    batch_size = args["batch_size"]
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(" *********************** Current dataloader length " + str(len(train_loader)) + " ***********************")

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    iteration = 0
    break_count = 0
    best_loss = 10000000
    stop_sign = 0

    for param in G_mapping.module.parameters():
        param.requires_grad = False
    for param in G_synthesis.module.parameters():
        param.requires_grad = False

    D, D_pair = get_all_D(args["dim"][0])
    D.cuda()
    D_pair.cuda()
    D, D_pair = nn.DataParallel(D), nn.DataParallel(D_pair)
    G = nn.DataParallel(Resnet(input_nc=args["dim"][2] + args["q_dim"], output_nc=3, nf=128).cuda())

    G_optimizer = get_optimizer(G, lr=1e-4)
    D_optimizer = get_optimizer(D, lr=1e-4)
    D_pair_optimizer = get_optimizer(D_pair, lr=1e-4)

    G_scheduler = get_scheduler(G_optimizer)
    D_scheduler = get_scheduler(D_optimizer)
    D_pair_scheduler = get_scheduler(D_pair_optimizer)

    for epoch in range(100):
        for latent_batch, q_batch in train_loader:
            latent_batch, q_batch = latent_batch.cuda(), q_batch.cuda()
            q_batch_upsampled = q_batch.unsqueeze(2).unsqueeze(3).repeat(1, 1, args["dim"][0], args["dim"][1])

            sharp_batch, fmap_batch = latent_to_image_stylegan2(
                G_mapping, G_synthesis, upsamplers, latent_batch, dim=args["dim"][1], return_upsampled_features=True, process_out=False
            )

            # print(latent_batch.shape, q_batch.shape, fmap_batch.shape, sharp_batch.shape, q_batch_upsampled.shape)
            # print(sharp_batch.min(), sharp_batch.max())

            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            D_pair_optimizer.zero_grad()

            deg_batch = G(torch.cat((fmap_batch, q_batch_upsampled), dim=1))

            # Maximize logits for generated images
            if iteration % 3 == 0:
                gen_logits = D(deg_batch, None)
                pair_logits = D_pair(torch.cat((sharp_batch, deg_batch), dim=1), None)
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) + 0.5 * torch.nn.functional.softplus(-pair_logits)
                loss_Gmain.mean().backward()
                G_optimizer.step()

            # Minimize logits for generated images
            if iteration % 3 == 1:
                gen_logits = D(deg_batch, None)
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                loss_Dgen.mean().backward()
                D_optimizer.step()

            if iteration % 3 == 2:
                real_deg_batch = get_real_deg(sharp_batch)
                real_pair_logits = D_pair(torch.cat((sharp_batch, real_deg_batch), dim=1), None)
                loss_Dpair_real = torch.nn.functional.softplus(-real_pair_logits) # -log(sigmoid(real_logits))
                loss_Dpair_real.mean().backward()
                D_pair_optimizer.step()

            iteration += 1
            if iteration % 10 == 0:
                print(f"Epoch: {epoch}, iter: {iteration}, loss(Gmain, Dgen, Dpair): {loss_Gmain.mean().item():.2f}, {loss_Dgen.mean().item():.2f}, {loss_Dpair_real.mean().item():.2f}")
            if iteration % 1000 == 0:
                torch.save(G.state_dict(), 'checkpoints/models/{iteration:04d}.pth')

    G_scheduler.step()
    D_scheduler.step()
    D_pair_scheduler.step()

    model_path = os.path.join(args["exp_dir"], "model_" + str(MODEL_NUMBER) + ".pth")
    torch.save({"model_state_dict": classifier.state_dict()}, model_path)
    print("Saved trained model to:", model_path)
    gc.collect()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--exp_dir", type=str, default="")
    parser.add_argument("--generate_data", type=bool, default=False)
    parser.add_argument("--save_vis", type=bool, default=False)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--num_sample", type=int, default=1000)
    args = parser.parse_args()

    opts = json.load(open(args.exp, "r"))
    print("Configuration: ", opts)

    if args.exp_dir != "":
        opts["exp_dir"] = args.exp_dir

    path = opts["exp_dir"]
    if os.path.exists(path):
        pass
    else:
        os.system("mkdir -p %s" % (path))
        print("Experiment folder created at: %s" % (path))

    os.system("cp %s %s" % (args.exp, opts["exp_dir"]))

    if args.generate_data:
        generate_data(opts, args.resume, args.num_sample, vis=args.save_vis, start_step=args.start_step)
    else:
        main(opts)
