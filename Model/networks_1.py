import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import pdb
import torch.nn.functional as F 

###############################################################################
# Functions
###############################################################################



class FBPCONVNet_big_drop(nn.Module):
    def __init__(self, input_nc=1, NFS=128, p=0.2):
        super(FBPCONVNet_big_drop, self).__init__()
        # create network model
        self.block_1_1 = None
        self.block_2_1 = None
        self.block_3_1 = None
        self.block_4_1 = None
        self.block_5 = None
        self.block_4_2 = None
        self.block_3_2 = None
        self.block_2_2 = None
        self.block_1_2 = None
        self.input_nc = input_nc
        self.NFS = NFS
        self.p = p
        self.create_model()
        

    def forward(self, input):
        
        block_1_1_output = self.block_1_1(input)
        block_1_1_output = self.block_1_1_drop(block_1_1_output)

        block_2_1_output = self.block_2_1(block_1_1_output)
        block_2_1_output = self.block_2_1_drop(block_2_1_output)

        block_3_1_output = self.block_3_1(block_2_1_output)
        block_3_1_output = self.block_3_1_drop(block_3_1_output)

        block_4_1_output = self.block_4_1(block_3_1_output)
        block_4_1_output = self.block_4_1_drop(block_4_1_output)

        block_5_output = self.block_5(block_4_1_output)
        block_5_output = self.block_5_drop(block_5_output)

        result = self.block_4_2(torch.cat((block_4_1_output, block_5_output), dim=1))
        result = self.block_4_2_drop(result)
        
        result = self.block_3_2(torch.cat((block_3_1_output, result), dim=1))
        result = self.block_3_2_drop(result)

        result = self.block_2_2(torch.cat((block_2_1_output, result), dim=1))
        result = self.block_2_2_drop(result)

        result = self.block_1_2(torch.cat((block_1_1_output, result), dim=1))
        result = self.block_1_2_drop(result)

        result = result + input
        return result

    def create_model(self):
        self.block_1_1_drop = nn.Dropout2d(p=self.p)
        self.block_2_1_drop = nn.Dropout2d(p=self.p)
        self.block_3_1_drop = nn.Dropout2d(p=self.p)
        self.block_4_1_drop = nn.Dropout2d(p=self.p)
        self.block_5_drop = nn.Dropout2d(p=self.p)
        self.block_4_2_drop = nn.Dropout2d(p=self.p)
        self.block_3_2_drop = nn.Dropout2d(p=self.p)
        self.block_2_2_drop = nn.Dropout2d(p=self.p)
        self.block_1_2_drop = nn.Dropout2d(p=self.p)
       
        kernel_size = 3
        padding = kernel_size // 2
        NFS = self.NFS 
        # block_1_1
        block_1_1 = []
        block_1_1.extend(self.add_block_conv(in_channels= self.input_nc, out_channels=NFS, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=NFS, out_channels=NFS, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=NFS, out_channels=NFS, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
                                         
        self.block_1_1 = nn.Sequential(*block_1_1)

        # block_2_1
        block_2_1 = [nn.MaxPool2d(kernel_size=2)]
        block_2_1.extend(self.add_block_conv(in_channels=NFS, out_channels=NFS*2, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_1.extend(self.add_block_conv(in_channels=NFS*2, out_channels=NFS*2, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
   
        self.block_2_1 = nn.Sequential(*block_2_1)

        # block_3_1
        block_3_1 = [nn.MaxPool2d(kernel_size=2)]
        block_3_1.extend(self.add_block_conv(in_channels=NFS*2, out_channels=NFS*4, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_1.extend(self.add_block_conv(in_channels=NFS*4, out_channels=NFS*4, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))

        self.block_3_1 = nn.Sequential(*block_3_1)

        # block_4_1
        block_4_1 = [nn.MaxPool2d(kernel_size=2)]
        block_4_1.extend(self.add_block_conv(in_channels=NFS*4, out_channels=NFS*8, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_1.extend(self.add_block_conv(in_channels=NFS*8, out_channels=NFS*8, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        
        self.block_4_1 = nn.Sequential(*block_4_1)

        # block_5
        block_5 = [nn.MaxPool2d(kernel_size=2)]
        block_5.extend(self.add_block_conv(in_channels=NFS*8, out_channels=NFS*16, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv(in_channels=NFS*16, out_channels=NFS*16, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv_transpose(in_channels=NFS*16, out_channels=NFS*8, kernel_size=kernel_size, stride=2,
                                                     padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_5 = nn.Sequential(*block_5)

        # block_4_2
        block_4_2 = []
        block_4_2.extend(self.add_block_conv(in_channels=NFS*16, out_channels=NFS*8, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(self.add_block_conv(in_channels=NFS*8, out_channels=NFS*8, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(
            self.add_block_conv_transpose(in_channels=NFS*8, out_channels=NFS*4, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_4_2 = nn.Sequential(*block_4_2)

        # block_3_2
        block_3_2 = []
        block_3_2.extend(self.add_block_conv(in_channels=NFS*8, out_channels=NFS*4, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(self.add_block_conv(in_channels=NFS*4, out_channels=NFS*4, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(
            self.add_block_conv_transpose(in_channels=NFS*4, out_channels=NFS*2, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_3_2 = nn.Sequential(*block_3_2)

        # block_2_2
        block_2_2 = []
        block_2_2.extend(self.add_block_conv(in_channels=NFS*4, out_channels=NFS*2, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(self.add_block_conv(in_channels=NFS*2, out_channels=NFS*2, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(
            self.add_block_conv_transpose(in_channels=NFS*2, out_channels=NFS, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_2_2 = nn.Sequential(*block_2_2)

        # block_1_2
        block_1_2 = []
        block_1_2.extend(self.add_block_conv(in_channels=NFS*2, out_channels=NFS, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=NFS, out_channels=NFS, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=NFS, out_channels=self.input_nc, kernel_size=1, stride=1,
                                             padding=0, batchOn=False, ReluOn=False))
        self.block_1_2 = nn.Sequential(*block_1_2)

    @staticmethod
    def add_block_conv(in_channels, out_channels, kernel_size, stride, padding, batchOn, ReluOn):
        seq = []
        # conv layer
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding)
        seq.append(conv)

        # batch norm layer
        batchOn=False
        if batchOn:
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            seq.append(batch_norm)
	
        # relu layer
        if ReluOn:
            seq.append(nn.ReLU())
        return seq

    @staticmethod
    def add_block_conv_transpose(in_channels, out_channels, kernel_size, stride, padding, output_padding, batchOn, ReluOn):
        seq = []

        convt = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, output_padding=output_padding)
        seq.append(convt)

        batchOn=False	
        if batchOn:
            batch_norm = nn.BatchNorm2d(num_features=out_channels)
            seq.append(batch_norm)
	
	
        if ReluOn:
            seq.append(nn.ReLU())
        return seq

