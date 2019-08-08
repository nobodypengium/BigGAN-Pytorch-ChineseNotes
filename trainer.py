import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from model_resnet import Generator, Discriminator
from utils import *


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel
        self.gpus = config.gpus

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        self.n_class = config.n_class
        self.chn = config.chn

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('build_model...')
        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def label_sampel(self):
        """
        生成选定类的度热编码
        :return:
        """
        label = torch.LongTensor(self.batch_size, 1).random_() % self.n_class
        one_hot = torch.zeros(self.batch_size, self.n_class).scatter_(1, label, 1)
        return label.squeeze(1).to(self.device), one_hot.to(self.device)

    def train(self):

        # Data iterator TODO:
        data_iter = iter(self.data_loader)  # data_loader中带有根据文件夹路径生成的编码，这里已经是一个图片对一个文件夹序号了
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model 只改变代数的记录，没有实际导入预训练模型
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        print('Start   ======  training...')
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== # 真的假的图片分开训练，真的数量=假的数量=batch size
            self.D.train()
            self.G.train()

            try:
                real_images, real_labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)  # 数据不够用了 再走一遍
                real_images, real_labels = next(data_iter)

            # Compute loss with real images

            real_labels = real_labels.to(self.device)
            real_images = real_images.to(self.device)

            d_out_real = self.D(real_images, real_labels)
            if self.adv_loss == 'wgan-gp':  # 根据D的预测值和真实的标签找loss
                d_loss_real = - torch.mean(d_out_real)  # 注意负号，虽然最后loss合起来是+，但是-放到这了
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = torch.randn(self.batch_size, self.z_dim).to(self.device)

            z_class, z_class_one_hot = self.label_sampel()  # 随机产生一些分类

            fake_images = self.G(z, z_class_one_hot)
            d_out_fake = self.D(fake_images, z_class)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize 反向传播的操作模板，这里只进行了没有惩罚的部分
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            #这里把惩罚的部分加上
            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty 梯度惩罚 TODO:这里是重点
                # 下面两行为从真实图片和虚假图片中间的分布中抽样一张图片
                alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device).expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out = self.D(interpolated)

                #检查抽样图片的梯度，因为损失函数比较复杂，应该保存之前的计算图，用之前的计算图来获取中间图片的梯度，用于计算惩罚
                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).to(self.device), # 全一向量是为了求一次导数
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0] #[0]这里提取的是矩阵部分，[1]是grad_fn=<MulBackward0>

                grad = grad.view(grad.size(0), -1) #(grad_size,1) 每个样本收缩为一列
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1)) #对每个样本求平方
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2) #所有样本平均到一个数

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step() #再补上惩罚的项的梯度的反向传播

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = torch.randn(self.batch_size, self.z_dim).to(self.device)
            z_class, z_class_one_hot = self.label_sampel()

            fake_images = self.G(z, z_class_one_hot) #Geerator接受多输入

            # Compute loss with fake images
            g_out_fake = self.D(fake_images, z_class)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean() #负号的意思是尽可能减少假图跟真图的差距
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward() #对G反向传播
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(
                    "Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_loss_fake: {:.4f}".
                    format(elapsed, step + 1, self.total_step, (step + 1),
                           self.total_step, d_loss_real.item(), d_loss_fake.item(), g_loss_fake.item()))

                if self.use_tensorboard: #这里再定义的tersorboard上写东西
                    self.writer.add_scalar('data/d_loss_real', d_loss_real.item(), (step + 1)) #.item()转换为标量 存储位置，x轴，y轴
                    self.writer.add_scalar('data/d_loss_fake', d_loss_fake.item(), (step + 1))
                    self.writer.add_scalar('data/d_loss', d_loss.item(), (step + 1))

                    self.writer.add_scalar('data/g_loss_fake', g_loss_fake.item(), (step + 1))
                    self.writer.flush()

            # Sample images
            if (step + 1) % self.sample_step == 0:
                print('Sample images {}_fake.png'.format(step + 1))
                fake_images = self.G(fixed_z, z_class_one_hot)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if (step + 1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):
        # code_dim=100, n_class=1000
        self.G = Generator(self.z_dim, self.n_class, chn=self.chn).to(self.device)
        self.D = Discriminator(self.n_class, chn=self.chn).to(self.device)
        if self.parallel:
            print('use parallel...')
            print('gpuids ', self.gpus)
            gpus = [int(i) for i in self.gpus.split(',')]

            self.G = nn.DataParallel(self.G, device_ids=gpus)
            self.D = nn.DataParallel(self.D, device_ids=gpus)

        # self.G.apply(weights_init)
        # self.D.apply(weights_init)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
                                            [self.beta1, self.beta2])  # 从G的参数中找出需要梯度回传的项
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr,
                                            [self.beta1, self.beta2])  # 从D的参数中找出需要梯度回传的项

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        """
        使用tensorboard存储log，这里之定义了writer，并没有实际指定写什么东西进去
        :return:
        """
        from tensorboardX import SummaryWriter
        # from logger import Logger
        # self.logger = Logger(self.log_path)

        tf_logs_path = os.path.join(self.log_path, 'tf_logs')
        self.writer = SummaryWriter(log_dir=tf_logs_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        """
        pytorch 似乎不会自动清空之前算的梯度，只会累加，所以每个batch需要手动清空
        :return:
        """
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        """
        解除归一化并输出某epoch选择的图片，做成雪碧图
        :param data_iter:
        :return:
        """
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
