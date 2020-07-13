import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
import argparse
import rawpy
from data_provider import *
from arch import Deep_Burst_Denoise
import torch.nn.functional as F
import torch.optim as optim
from generate_list import generate_list
from PIL import Image

TENSOR_BOARD = False
if TENSOR_BOARD:
    from tensorboardX import SummaryWriter

class sampler(Sampler):
    def __init__(self, batch_size, data_source):
        super(sampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.total_size = len(data_source)

    def __iter__(self):
        if self.total_size % self.batch_size == 0:
            return iter(torch.randperm(self.total_size))
        else:
            return iter(torch.randperm(self.total_size).tolist() + torch.randperm(self.total_size).tolist()[:self.batch_size-self.total_size % self.batch_size])

    def __len__(self):
        return self.total_size


def adjust_learning_rate(optimizer, lr):

    lr = lr / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def rnn_fcn_train(dataset_path, txt_file, batch_size=4, patch_size=512, lr=5e-4, lr_decay=1000, max_epoch=10000):
    dataset = Data_Provider(dataset_path, txt_file, patch_size=patch_size, train=True)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler(batch_size, dataset),
        num_workers= 4
    )




    model = Deep_Burst_Denoise(1, 3).cuda()

    # switch to the train mode
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    
    file_name = "model_newest_700.pkl"
    if os.path.exists(file_name):
        model.load_state_dict(torch.load("/home/Diplom/Recurrent-Fully-Convolutional-Networks/"+file_name))
        print("-------------------Get " +file_name+"-------------------")

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if TENSOR_BOARD:
        summary = SummaryWriter('./logs', comment='loss')

    loss_file = open('./loss_logs/loss_log.txt', 'w+')

    global_step = 0

    min_loss = 10**9+7
    try:
        for epoch in range(max_epoch):

            if epoch > 0 and epoch % lr_decay == 0:
                lr = adjust_learning_rate(optimizer, lr)
            print('=============Epoch:{}, lr:{}.============'.format(epoch+1, lr))
            for step, (train_data, gt, _) in enumerate(data_loader):
                train_data = train_data.cuda()
                gt = gt.cuda()
                loss_temp = 0
                for channel in range(train_data.size(1)):
                    if channel == 0:
                        sfn_out, mfn_out, mfn_f = model(train_data[:, channel, ...].unsqueeze(1))
                    else:
                        sfn_out, mfn_out, mfn_f = model(train_data[:, channel, ...].unsqueeze(1), mfn_f)

                    loss_temp += F.l1_loss(sfn_out, gt) + F.l1_loss(mfn_out, gt)

                global_step += 1
                if TENSOR_BOARD:
                    summary.add_scalar('loss', loss_temp, global_step)

                print('Epoch:{}, Step:{}, Loss:{:.4f}.'.format(epoch+1, step, loss_temp))

                optimizer.zero_grad()
                loss_temp.backward()
                optimizer.step()

                loss_file.write('{},'.format(loss_temp))

                if loss_temp < min_loss:
                    min_loss = loss_temp
                    torch.save(model.state_dict(), 'model_min_loss.pkl')
                if global_step % 1000 == 0:
                    torch.save(model.state_dict(), 'model_newest.pkl')
    finally:
        loss_file.close()

if __name__ == '__main__':
    dataset_path = '/home/Diplom/Recurrent-Fully-Convolutional-Networks/Sony/Sony'
    generate_list(dataset_path, './', burst=8)
    rnn_fcn_train(
                dataset_path=dataset_path,
                txt_file='./train_list.txt',
                batch_size=1,
                patch_size=272,
                lr=5e-5
    ) 
