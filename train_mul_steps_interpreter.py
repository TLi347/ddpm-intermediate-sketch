import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,6,"
import gc
import numpy as np 

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions, pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform, ImageDataset
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.script_util_4ddpmseg import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.dist_util import dev

from torchvision.utils import save_image

# x = torch.from_numpy(np.random.uniform(0,100,size=(8,2048))).to(dtype=torch.float32).cuda()
# x.requires_grad=True
# target = torch.ones_like(x)*5
# target.requires_grad=True
# model = nn.Sequential( 
#         nn.Linear(2048, 2048),
#         nn.ReLU(),
#         nn.Linear(2048, 2048),
#         nn.ReLU(),
# ).cuda()
# loss_fn = nn.MSELoss()
# opt = torch.optim.Adam(model.parameters(), lr=0.1)

# model.train()
# for i in range(200):
#     pred = model(x)
#     loss = loss_fn(pred,target)
#     print(i, ' loss = ', loss.mean().item())
#     loss.backward()
#     opt.step()
    
#     if i>9 and i%10==0 and i<50:
#         with torch.no_grad():
#             for ii in range(5):
#                 pred = model(x)
#                 loss = loss_fn(pred,target)
#                 print(ii, 'no grad  ', i, ' loss = ', loss.mean().item())

# print(model(x)[:2,:5])
# print(target[:2,:5])


def prepare_dataset(args,split="train"):
    feature_extractor = create_feature_extractor(**args)
    
    if split=="train":
        print(f"Preparing the train set for {args['category']}...")
        dataset = ImageDataset(
            data_dir=args['training_path'],
            resolution=args['image_size'],
            
        )
        sample_num = len(dataset)*len(args["steps"])
    elif split=="test":
        print(f"Preparing the test set for {args['category']}...")
        dataset = ImageDataset(
            data_dir=args['testing_path'],
            resolution=args['image_size'],
            
        )
        sample_num = len(dataset)*len(args["steps"])
    else:
        print(f"Preparing the validation set for {args['category']}...")
        dataset = ImageDataset(
            data_dir=args['validation_path'],
            resolution=args['image_size'],
            
        )
        sample_num = args['validation_sample_num']*len(args["steps"])
    
    X = torch.zeros((sample_num, *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((sample_num, *args['edge_dim'][::-1]), dtype=torch.float)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    from sklearn.cluster import KMeans
    from skimage.io import imsave
    
    sample_idx = 0
    for row, (img, edge_img, txt) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
        features_batch = feature_extractor(img, noise=noise)
        for features in features_batch:
            X[sample_idx] = collect_features(args, features).cpu()
            edge_img = edge_img[None].to(dev())
            y[sample_idx] = edge_img
            
            
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(X[sample_idx].view(3,-1).permute(1,0))
            y_kmeans = kmeans.predict(X[sample_idx].view(3,-1).permute(1,0))
            samve_t=args['steps']
            imsave(f'./pixel_classifiers/horse_21/dim=120/time={samve_t}.png',y_kmeans.reshape(120,120))
            
            sample_idx += 1
            
            print(X[sample_idx].shape, y[sample_idx].shape)
            
        if sample_idx+1>=sample_num:
            break
        
    print(' X 1 = ', X.shape, y.shape)
    d = X.shape[1]
    edge_d = y.shape[1]
    print(f'Total dimension {d}')
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.permute(1,0,2,3).reshape(edge_d, -1).permute(1, 0)
    print(' X 2 = ', X.shape, y.shape, X.dtype, y.dtype)
    return X, y

def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        
    )

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    preds, gts, uncertainty_scores = [], [], []
    for img, edge_img, txt in tqdm(dataset):        
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)
        
        edge_img = edge_img[None]
        label = edge_img
        label = label.view(args['edge_dim'][-1], -1).permute(1, 0)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=[args['dim'][:-1][0], args['dim'][:-1][1], 3]
        )
        label = (label.clamp(0,1)*255).to(torch.int8)
        gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())
    
    save_predictions(args, dataset.image_paths, gts, fp_name='ref')
    save_predictions(args, dataset.image_paths, preds)
    # miou = compute_iou(args, preds, gts)
    # print(f'Overall mIoU: ', miou)
    # print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    features, labels = prepare_dataset(args, split="train")
    train_data = FeatureDataset(features, labels)

    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    print(f" *********************** Current number data {len(features)} ***********************")

    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    
    
    features, labels = prepare_dataset(args ,split="val")
    validation_data = FeatureDataset(features, labels)
    print(" *********************** Current validation dataloader length " +  str(len(train_loader)) + " ***********************")
    
    # for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):

    #     gc.collect()
    #     classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])
    #     classifier.init_weights()

    #     classifier = nn.DataParallel(classifier).cuda()
    #     # criterion = nn.CrossEntropyLoss()
    #     criterion = nn.MSELoss()
    #     optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    #     classifier.train()

    #     iteration = 0
    #     break_count = 0
    #     best_loss = 10000000
    #     stop_sign = 0
    #     for epoch in range(500):
    #         for X_batch, y_batch in train_loader:
    #             X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())

    #             optimizer.zero_grad()
    #             y_pred = classifier(X_batch)
    #             loss = criterion(y_pred, y_batch)
    #             # acc = multi_acc(y_pred, y_batch)

    #             loss.backward()
    #             optimizer.step()

    #             iteration += 1
    #             if iteration % 1000 == 0:
    #                 print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item())
    #             log_wirter.add_scalar(f'{MODEL_NUMBER}/train_loss', loss, global_step=iteration)
                
    #             if iteration % 5000 == 0 or iteration<2:
    #                 print('Epoch : ', str(epoch), 'iteration', iteration, ' Validation starting...')
    #                 preds, gts, suc_gts, uncertainty_scores = [], [], [], []
    #                 for validation_batch_id in range(args['validation_sample_num']*len(args["steps"])):
    #                     validation_batch_size = 120**2
    #                     validation_X_batch = validation_data[:][0][validation_batch_size*validation_batch_id:validation_batch_size*(validation_batch_id+1)].to(dev())
    #                     validation_y_batch = validation_data[:][1][validation_batch_size*validation_batch_id:validation_batch_size*(validation_batch_id+1)].to(dev())
                        
    #                     with torch.no_grad():
    #                         pred = classifier(validation_X_batch)
    #                         validation_loss = criterion(pred, validation_y_batch)
    #                         pred = pred.reshape([120,120,3]).permute(2,0,1)
                        
    #                     gts.append(validation_y_batch.reshape([120,120,3]).permute(2,0,1))
    #                     preds.append(pred)
                    
    #                 ref = torch.cat(gts,dim=2).clamp(0,1).cpu().detach().numpy()
    #                 rec = torch.cat(preds,dim=2).clamp(0,1).cpu().detach().numpy()
    #                 log_wirter.add_image(f'{MODEL_NUMBER}/eva_ref', ref, global_step=int(iteration//5000))
    #                 log_wirter.add_image(f'{MODEL_NUMBER}/eva_rec', rec, global_step=int(iteration//5000))
    #                 log_wirter.add_scalar(f'{MODEL_NUMBER}/val_loss',validation_loss,global_step=int(iteration//5000))
                    

    #             if epoch > 3:
    #                 if loss.item() < best_loss:
    #                     best_loss = loss.item()
    #                     # model_path = os.path.join(args['exp_dir'], 'model_' + str(MODEL_NUMBER) + '_tmp.pth')
    #                     # print('save to:',model_path)
    #                     # torch.save({'model_state_dict': classifier.state_dict()}, model_path)
    #                     break_count = 0
    #                 else:
    #                     break_count += 1

    #                 if break_count > 80:
    #                     stop_sign = 1
    #                     print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
    #                     break

    #         if stop_sign == 1:
    #             break

    #     model_path = os.path.join(args['exp_dir'], 
    #                               'model_' + str(MODEL_NUMBER) + '.pth')
    #     MODEL_NUMBER += 1
    #     print('save to:',model_path)
    #     torch.save({'model_state_dict': classifier.state_dict()},
    #                model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--validation_sample_num', type=int, default=4)
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str, default="experiments/horse_21/mul_steps_ddpm.json")
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = 256#opts['dim'][0]

    
    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)
    log_wirter = SummaryWriter(opts['exp_dir'])

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train(opts)
    
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)
