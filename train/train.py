import argparse
import logging
import sys
import os
import numpy as np
import time
import copy
import random

os.environ['QT_QPA_PLATFORM']='offscreen'

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

# import matplotlib; matplotlib.use('agg')

sys.path.append('./')
from nyuv2.nyuv2_raw_loader import NYUV2RawLoader
from module.d3 import D3
from utils.log_helper import init_log, print_speed
from utils.save_load import load_pretrain, restore_from, save_checkpoint
from utils.nn_fill import NN_fill, generate_mask

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--dataset_dir', default="/home/zxm/dataset/dataset/nyuv2/nyu_depth_v2", help='dataset directory ')
parser.add_argument('--save_dir', default='./checkpoints', help='directory to save models')
parser.add_argument('--summary_dir', default='./runs', help='directory to save summary, can load by tensorboard')
parser.add_argument('--resume', default='./checkpoints/model_best.pth', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epoches', default=20, type=int, metavar='N', help='number of total epoches to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr_decay', '--lr_decay', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--lr_decay_step', default=8, type=float, help='learning rate decay every 8 epoches')
parser.add_argument('--test_rate', default=1, type=int, help='run test every test_rate epoches')
parser.add_argument('--log_rate', default=1, type=int, help='save model every log_rate epoches')

args = parser.parse_args()

writer = SummaryWriter(log_dir=args.summary_dir)

# deterministic
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# fix random seed
random.seed(1)
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class NYU_Dataset(Dataset):
    def __init__(self, dataset_dir, mode='train', transform=None, mask_grid=24):
        assert mode == 'train' or mode == 'test'

        self.raw_dataloader = NYUV2RawLoader(dataset_dir, mode)
        self.transform = transform
        img, _ = self.raw_dataloader.get_one_example()
        self.mask = generate_mask(mask_grid, mask_grid, img.shape[0], img.shape[1])

    def __len__(self):
        return self.raw_dataloader.total_samples

    def __getitem__(self, idx):
        img, depth_prj = self.raw_dataloader.get_one_example(idx)
        s1, s2 = NN_fill(img, depth_prj, self.mask)

        sample = {'rgb': img,
                  'depth': depth_prj,
                  's1': s1,
                  's2': s2}

        if self.transform:
            sample = self.transform(sample)

        return sample


class HFlips(object):
    """horizontal flips"""

    def __call__(self, sample):
        p = random.random()
        if p > 0.5:
            sample["rgb"] = np.fliplr(sample['rgb'])
            sample["depth"] = np.fliplr(sample['depth'])
            sample["s1"] = np.fliplr(sample['s1'])
            sample["s2"] = np.fliplr(sample['s2'])
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'rgb': torch.from_numpy(sample['rgb'].copy()),  #.copy() to avoid neagtive stride ValueError
                'depth': torch.from_numpy(sample['depth'].copy()),
                's1': torch.from_numpy(sample['s1'].copy()),
                's2': torch.from_numpy(sample['s2'].copy())}


@torch.no_grad()
def compute_errors(gt, pred):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt) / torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


def l2_loss(gt, pred):
    loss, total_points = 0, 0
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        total_points = total_points + torch.sum(valid).type(torch.cuda.FloatTensor)
        loss = loss + torch.sum((valid_gt - valid_pred) * (valid_gt - valid_pred))

    loss = loss / total_points
    return loss


def build_nyu_dataloader(dataset_dir):
    logger = logging.getLogger('global')
    logger.info("Build dataloader...")

    trans = transforms.Compose([HFlips(), ToTensor()])

    train_nyu_dataset = NYU_Dataset(dataset_dir, "train", trans)
    train_nyu_loader = DataLoader(train_nyu_dataset, batch_size=args.batch_size, num_workers=args.workers)

    test_nyu_dataset = NYU_Dataset(dataset_dir, "test")
    test_nyu_loader = DataLoader(test_nyu_dataset, batch_size=args.batch_size, num_workers=args.workers)


    logger.info('Build nyu dataloader done')

    return train_nyu_loader, test_nyu_loader


def main():
    # init logger
    init_log('global', args.save_dir, logging.INFO)
    logger = logging.getLogger('global')
    # print arguments
    for arg in vars(args):
        logger.info("{}: {}".format(arg, getattr(args, arg)))

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build dataloader and model
    train_loader, test_loader = build_nyu_dataloader(args.dataset_dir)
    opts = {"L": 5, "k": 12, "bn": True}
    model = D3(opts)

    # check GPU numbers and deploy parallel
    # parallel = False
    # if torch.cuda.device_count() > 1:
    #     parallel = True
    #     logger.info("Let's use {:d} GPUs!".format(torch.cuda.device_count()))
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)
    model.to(device)

    logger.info("*" * 40)
    logger.info(model)
    logger.info("*" * 40)

    # optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         model, _, args.start_epoch = restore_from(model, optimizer, args.resume)

    # set the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_abs_rel = 0.0
    logger.info("Start training...")

    # epoches = args.batches // train_loader.__len__()

    for epoch in range(args.epoches):

        for g in optimizer.param_groups:
            g['lr'] = args.lr * (1 - args.lr_decay) ** (epoch // args.lr_decay_step)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        t0 = time.time()
        train_one_epoch(train_loader, model, optimizer, device, epoch)
        t1 = time.time()

        if epoch % args.test_rate == 0:
            test_abs_rel = test_one_epoch(test_loader, model, device, epoch)
            if test_abs_rel < best_abs_rel:
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.cuda.empty_cache()

        if epoch % args.test_rate == 0:
            filename = os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch + 1))
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                is_best=False,
                filename=filename
            )
            logger.info("Saved model : {}".format(filename))

        print_speed(epoch, t1 - t0, args.epoches)

        save_checkpoint(
            {
                'batch_num': epoch,
                'state_dict': best_model_wts,
                'optimizer': optimizer.state_dict()
            },
            is_best=True,
            filename=os.path.join(args.save_dir, 'model_best.pth')
        )

    writer.close()


def train_one_epoch(train_loader, model, optimizer, device, epoch):
    logger = logging.getLogger('global')
    model.train()

    lossitem, abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0
    num_batches = 0.0

    for i_batch, sample_batched in enumerate(train_loader):
        t0 = time.time()

        rgb = sample_batched['rgb'].type(torch.FloatTensor)
        depth = sample_batched['depth']
        s1 = sample_batched['s1'].type(torch.FloatTensor)
        s2 = sample_batched['s2'].type(torch.FloatTensor)

        b = rgb.shape[0]

        rgb = rgb.to(device)
        depth = depth.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)

        s1.unsqueeze_(-1)
        s2.unsqueeze_(-1)

        s1s2 = torch.cat((s1, s2), 3)

        rgb = rgb.permute(0,3,1,2)
        s1s2 = s1s2.permute(0,3,1,2)

        # zero the parameter gradients
        optimizer.zero_grad()

        depth_predict = model(rgb, s1s2)

        depth_predict.squeeze_(1)
        loss = l2_loss(depth, depth_predict)

        # backward + optimize
        loss.backward()
        optimizer.step()

        lossitem0 = loss.item()
        abs_diff0, abs_rel0, sq_rel0, a10, a20, a30 = compute_errors(depth, depth_predict)

        num_batches += 1
        lossitem += lossitem0
        abs_diff += abs_diff0
        abs_rel += abs_rel0
        sq_rel += sq_rel0
        a1 += a10
        a2 += a20
        a3 += a30

        t1 = time.time()
        print_speed(i_batch, t1 - t0, train_loader.__len__())

    lossitem = lossitem / num_batches
    abs_diff = abs_diff / num_batches
    abs_rel = abs_rel / num_batches
    sq_rel = sq_rel / num_batches
    a1 = a1 / num_batches
    a2 = a2 / num_batches
    a3 = a3 / num_batches

    logger.info('Train Loss: {:.4f},'.format(lossitem))
    logger.info('                    abs_diff: {:.4f}'.format(abs_diff))
    logger.info('                    abs_rel: {:.4f}'.format(abs_rel))
    logger.info('                    sq_rel: {:.4f}'.format(sq_rel))
    logger.info('                    a1: {:.4f}'.format(a1))
    logger.info('                    a2: {:.4f}'.format(a2))
    logger.info('                    a3: {:.4f}'.format(a3))

    writer.add_scalar('train/loss', lossitem, epoch)
    writer.add_scalar('train/abs_diff', abs_diff, epoch)
    writer.add_scalar('train/abs_rel', abs_rel, epoch)
    writer.add_scalar('train/sq_rel', sq_rel, epoch)
    writer.add_scalar('train/a1', a1, epoch)
    writer.add_scalar('train/a2', a2, epoch)
    writer.add_scalar('train/a3', a3, epoch)

def test_one_epoch(train_loader, model, device, epoch):
    logger = logging.getLogger('global')
    model.eval()

    lossitem, abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0
    num_batches = 0.0

    for i_batch, sample_batched in enumerate(train_loader):
        rgb = sample_batched['rgb'].type(torch.FloatTensor)
        depth = sample_batched['depth']
        s1 = sample_batched['s1'].type(torch.FloatTensor)
        s2 = sample_batched['s2'].type(torch.FloatTensor)

        rgb = rgb.to(device)
        depth = depth.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)

        s1.unsqueeze_(-1)
        s2.unsqueeze_(-1)

        s1s2 = torch.cat((s1, s2), 3)

        rgb = rgb.permute(0, 3, 1, 2)
        s1s2 = s1s2.permute(0, 3, 1, 2)

        depth_predict = model(rgb, s1s2)
        depth_predict.squeeze_(1)

        loss = l2_loss(depth, depth_predict)

        lossitem0 = loss.item()
        abs_diff0, abs_rel0, sq_rel0, a10, a20, a30 = compute_errors(depth, depth_predict)

        num_batches += 1
        lossitem += lossitem0
        abs_diff += abs_diff0
        abs_rel += abs_rel0
        sq_rel += sq_rel0
        a1 += a10
        a2 += a20
        a3 += a30

    lossitem = lossitem / num_batches
    abs_diff = abs_diff / num_batches
    abs_rel = abs_rel / num_batches
    sq_rel = sq_rel / num_batches
    a1 = a1 / num_batches
    a2 = a2 / num_batches
    a3 = a3 / num_batches

    logger.info('Test  Loss: {:.4f},'.format(lossitem))
    logger.info('                    abs_diff: {:.4f}'.format(abs_diff))
    logger.info('                    abs_rel: {:.4f}'.format(abs_rel))
    logger.info('                    sq_rel: {:.4f}'.format(sq_rel))
    logger.info('                    a1: {:.4f}'.format(a1))
    logger.info('                    a2: {:.4f}'.format(a2))
    logger.info('                    a3: {:.4f}'.format(a3))

    writer.add_scalar('test/loss', lossitem, epoch)
    writer.add_scalar('test/abs_diff', abs_diff, epoch)
    writer.add_scalar('test/abs_rel', abs_rel, epoch)
    writer.add_scalar('test/sq_rel', sq_rel, epoch)
    writer.add_scalar('test/a1', a1, epoch)
    writer.add_scalar('test/a2', a2, epoch)
    writer.add_scalar('test/a3', a3, epoch)

    return abs_rel

if __name__ == "__main__":
    main()
