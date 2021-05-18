import numpy as np
import pandas as pd
import tables as tb
import torch
import sys
import sparseconvnet as scn
from .data_loaders import DataGen, collatefn, LabelType
from next_sparseconvnet.networks.architectures import UNet
from torch.utils.tensorboard import SummaryWriter

from invisible_cities.io.dst_io import df_writer
from invisible_cities.cities.components import index_tables

def IoU(true, pred, nclass = 3):
    """
        Intersection over union is a metric for semantic segmentation.
        It returns a IoU value for each class of our input tensors/arrays.
    """
    eps = sys.float_info.epsilon
    confusion_matrix = np.zeros((nclass, nclass))

    for i in range(len(true)):
        confusion_matrix[true[i]][pred[i]] += 1

    IoU = []
    for i in range(nclass):
        IoU.append((confusion_matrix[i, i] + eps) / (sum(confusion_matrix[:, i]) + sum(confusion_matrix[i, :]) - confusion_matrix[i, i] + eps))
    return np.array(IoU)


def train_one_epoch_segmentation(epoch_id, net, criterion, optimizer, loader, nclass = 3):
    """
        Trains the net for all the train data one time
    """
    net.train()
    loss_epoch, iou_epoch = 0, np.zeros(nclass)
    for batchid, (coord, ener, label, event) in enumerate(loader):
        batch_size = len(event)
        ener, label = ener.cuda(), label.cuda()

        optimizer.zero_grad()

        output = net.forward((coord, ener, batch_size))

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()

        #IoU
        softmax = torch.nn.Softmax(dim = 1)
        prediction = torch.argmax(softmax(output), 1)
        iou_epoch += IoU(label.cpu(), prediction.cpu(), nclass = nclass)

    loss_epoch = loss_epoch / len(loader)
    iou_epoch = iou_epoch / len(loader)
    epoch_ = f"Train Epoch: {epoch_id}"
    loss_ = f"\t Loss: {loss_epoch:.6f}"
    print(epoch_ + loss_)

    return loss_epoch, iou_epoch


def valid_one_epoch_segmentation(net, criterion, loader, nclass = 3):
    """
        Computes loss and IoU for all the validation data
    """
    net.eval()
    loss_epoch, iou_epoch = 0, np.zeros(nclass)
    with torch.autograd.no_grad():
        for batchid, (coord, ener, label, event) in enumerate(loader):
            batch_size = len(event)
            ener, label = ener.cuda(), label.cuda()

            output = net.forward((coord, ener, batch_size))

            loss = criterion(output, label)

            loss_epoch += loss.item()

            #IoU
            softmax = torch.nn.Softmax(dim = 1)
            prediction = torch.argmax(softmax(output), 1)
            iou_epoch += IoU(label.cpu(), prediction.cpu())

        loss_epoch = loss_epoch / len(loader)
        iou_epoch = iou_epoch / len(loader)
        loss_ = f"\t Validation Loss: {loss_epoch:.6f}"
        print(loss_)

    return loss_epoch, iou_epoch



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



def train_segmentation(*,
                       nepoch,
                       train_data_path,
                       valid_data_path,
                       train_batch_size,
                       valid_batch_size,
                       net,
                       criterion,
                       optimizer,
                       checkpoint_dir,
                       tensorboard_dir,
                       num_workers,
                       nevents_train = None,
                       nevents_valid = None):
    """
        Trains the net nepoch times and saves the model anytime the validation loss decreases
    """
    train_gen = DataGen(train_data_path, LabelType.Segmentation, nevents = nevents_train)
    valid_gen = DataGen(valid_data_path, LabelType.Segmentation, nevents = nevents_valid)

    loader_train = torch.utils.data.DataLoader(train_gen,
                                               batch_size = train_batch_size,
                                               shuffle = True,
                                               num_workers = num_workers,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = False)
    loader_valid = torch.utils.data.DataLoader(valid_gen,
                                               batch_size = valid_batch_size,
                                               shuffle = True,
                                               num_workers = 1,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = False)

    start_loss = np.inf
    writer = SummaryWriter(tensorboard_dir)
    for i in range(nepoch):
        train_loss, train_iou = train_one_epoch_segmentation(i, net, criterion, optimizer, loader_train)
        valid_loss, valid_iou = valid_one_epoch_segmentation(net, criterion, loader_valid)

        if valid_loss < start_loss:
            save_checkpoint({'state_dict': net.state_dict(),
                             'optimizer': optimizer.state_dict()}, f'{checkpoint_dir}/net_checkpoint_{i}.pth.tar')
            start_loss = valid_loss

        writer.add_scalar('loss/train', train_loss, i)
        for k, iou in enumerate(train_iou):
            writer.add_scalar(f'iou/train_{k}class', iou, i)

        writer.add_scalar('loss/valid', valid_loss, i)
        for k, iou in enumerate(valid_iou):
            writer.add_scalar(f'iou/valid_{k}class', iou, i)

        writer.flush()
    writer.close()



def predict_segmentation(data_path, output_name, net, batch_size, nevents):
    gen    = DataGen(data_path, LabelType.Segmentation, nevents = nevents)
    loader = torch.utils.data.DataLoader(gen,
                                         batch_size = batch_size,
                                         shuffle = False,
                                         num_workers = 1,
                                         collate_fn = collatefn,
                                         drop_last = False,
                                         pin_memory = False)

    net.eval()
    start_id = 0
    softmax = torch.nn.Softmax(dim = 1)
    with torch.autograd.no_grad():
        for batchid, (coord, ener, label, event) in enumerate(loader):
            batch_size = len(event)
            ener, label = ener.cuda(), label.cuda()
            output = net.forward((coord, ener, batch_size))
            y_pred = softmax(output).cpu().detach().numpy()

            nclasses = y_pred.shape[1]
            pd_dict = {f'pred_{i}':y_pred[:, i] for i in range(nclasses)}
            cords_name = ['X', 'Y', 'Z']
            pd_dict.update ({cords_name[i]:coord[:, i].cpu().detach().numpy() for i in range(3)})
            pd_dict.update({'energy':ener.cpu().detach().numpy().flatten(), 'label':label.cpu().detach().numpy().flatten()})
            pd_dict.update({'dataset_id':start_id+coord[:, -1].cpu().detach().numpy()})
            df = pd.DataFrame.from_records(pd_dict)
            with tb.open_file(output_name, 'a') as h5out:
                df_writer(h5out, df, 'DATASET', 'VoxelsPred', columns_to_index=['dataset_id'])
            start_id += batch_size
    index_tables(output_name)
