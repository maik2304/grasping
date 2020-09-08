import argparse
import datetime
import json
import logging
import os
import sys

import random
import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
import torch.nn as nn
import torchvision.models as models



from utils.data.cornell_data import CornellDataset
from model.model import set_parameter_requires_grad,initialize_model,MyResNet
from utils.dataset_processing import grasp, image

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='alexnet',
                        help='Network name in inference/models')
    parser.add_argument('--criterion', type=str, default='MSE',
                        help='metric error to evaluate the error')
    # Datasets
    parser.add_argument('--path', type=str,default='Dataset',
                        help='Path to dataset')       
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')    
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')    
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='Weigth decay')
    
    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')

    args = parser.parse_args()
    
    return args

def training(epoch, net, device, criterion, train_data, optimizer,tb):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param criterion: function to evaluate the loss
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    
    results = 0
    batch_idx = 0
    
    # change flag of training
    net.train()
    
    for sample in train_data:        
        
        batch_idx +=1

        # load of the normalized image
        img = sample['img'].to(device)
        
        # load of the ground bounding boxes
        ground_bb = sample['bb'].to(device)
        
        # forward propagation
        pred_bb = net(img)
        
        # evaluate the loss
        loss = criterion(ground_bb,pred_bb)
        
        results += loss.item()
        
        #  gradients are zeroed
        optimizer.zero_grad()
        
        # backward propagation
        loss.backward()
        
        # optimization of the parameters
        optimizer.step()        

        logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss))
    
    return results/batch_idx
    
    
def validate(net, device, val_loader,criterion):
    
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_loader: Validation Dataset
    :return: results: correct,failed,validation loss
    
    """
    
    net.eval()
    
    results = {
        'correct': 0,
        'failed': 0,
        'loss':0        
    }
    
    batch_idx = 0
    
    with torch.no_grad():
        
        for sample in val_loader:
            
            batch_idx +=1

            # load of the normalized image
            img = sample['img'].to(device)            
            
            # load of the ground bounding boxes
            ground_bb = sample['bb'].to(device).squeeze().cpu()
            
            # forward propagation
            pred_bb = net(img).to(device).squeeze().cpu()
            
            # evaluate the validation loss
            loss = criterion(ground_bb,pred_bb)
            
            # grasp.Grasp is a class with __init__(self,center,angle,length,width)
            ground_gr = grasp.Grasp((ground_bb[0],ground_bb[1]),np.arctan2(ground_bb[2],ground_bb[3])/2,ground_bb[4],ground_bb[5])
            ground_rect = ground_gr.as_gr
            
            pred_gr = grasp.Grasp((pred_bb[0],pred_bb[1]),np.arctan2(pred_bb[2],pred_bb[3])/2,pred_bb[4],pred_bb[5])
            pred_rect = pred_gr.as_gr
            
            iou_param = ground_rect.iou(pred_rect)
            
            if iou_param > 0.25:
                results['correct'] += 1
            else:
                results['failed'] += 1

            results['loss'] += loss.item()
    
    results['loss'] /= batch_idx

    logging.info('Loss: {:0.4f}'.format(results['loss']))

    return results

def run():
    
    args = parse_args()
    
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    
    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)
    
    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)
            
    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set the seed for reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

     # Load the network    
    logging.info('Loading Network...')
    
    # Initialization of the network
    net = initialize_model(args.network, 6, feature_extract=False, use_pretrained=True)
    net = net.to(device)
    
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr)
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay = args.wd)
        # Decay LR by a factor of 0.1 every 7 epochs
        #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))
    
    if args.criterion.lower() == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError('Criterion {} is not implemented'.format(args.criterion))
    
    logging.info('Done')  
    
     # Print model architecture.
    summary(net, (3, 224, 224))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (3, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()
    
    # Load Dataset
    logging.info('Loading Cornell Dataset...')
        
    dataset = CornellDataset(args.path)
    
    logging.info('Done')  
    
    logging.info('Dataset size is {}'.format(dataset.len))       
    
    count = 0
    best_iou = 0.0

    indices = list(range(dataset.len))
    split = int(np.floor(0.9 * dataset.len))
    #np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True         
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=True
    )       
    
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))
    
    for epoch in range(args.epochs):
        
        count+=1
        
        np.random.seed(20)
        random.seed(20) 

        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = training(epoch, net, device, criterion, train_loader, optimizer,tb)
        #scheduler.step()        
        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results,count)
        
        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_loader,criterion)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                    test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        
        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), count)
        tb.add_scalar('loss/val_loss', test_results['loss'], count)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou
            
if __name__ == '__main__':
    run()