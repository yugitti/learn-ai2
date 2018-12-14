import torch
import torch.nn.functional as F
import os
import torchvision.utils as vutils
import logger as log
import numpy as np
import os

class NNRun(object):

    def __init__(self, netD, netG, optimizerD, optimizerG, criterionD, criterionG, device, fixed_noise, logger, args):

        self.netD = netD
        self.netG = netG
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.criterionD = criterionD
        self.criterionG = criterionG
        self.nz = args.nz
        self.epochs = args.epochs

        self.device = device

        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.iters = args.iters
        # self.logger = log.TrainLogger(args)

        self.feature_matching = args.feature_matching
        self.minibatch = args.mini_batch

        self.fixed_noise = fixed_noise
        self.save_dir = args.save_dir
        self.prefix = args.prefix
        self.log_interval = args.log_interval
        self.logger = logger

    def train(self, label_loader, unlabel_loader, test_loader):

        # real dataとface dataのラベルを作成
        real_label = 1
        fake_label = 0
        correct = 0
        total = 0
        loss_acc = 0

        self.netD.train()
        self.netG.train()

        # for epoch in range(self.epochs):
        #     for i, data in enumerate(dataloader, 0):
        for iter_id in range(1, self.iters + 1):


            label_data, label_target = next(label_loader)
            label_data, label_target = label_data.to(self.device), label_target.to(self.device)

            unlabel_data, _ = next(unlabel_loader)
            unlabel_data = unlabel_data.to(self.device)

            batch_size = unlabel_data.size(0)

            self.netD.zero_grad()
            ''' #############  Discriminator の supervised dataを学習 [real dataのみ] #############'''
            # label = torch.full((batch_size,), real_label, device=self.device)
            outputD_label, _ = self.netD(label_data, self.feature_matching)
            errD_label = self.criterionD(outputD_label, label_target)

            ''' #############  Discriminator の unsupervised dataを学習 [real data]  ############# '''

            outputD_unlabel_real, feature_unlabel_real = self.netD(unlabel_data, self.feature_matching)
            # log_real_z = self.log_sum_exp(outputD_unlabel_real, 1)
            log_real_z = torch.logsumexp(outputD_unlabel_real, 1)
            log_real_d = log_real_z - F.softplus(log_real_z, 1)
            errD_unlabel_real = -torch.mean(log_real_d, 0)
            # errD_unlabel_real = -(torch.mean(log_real_z) - torch.mean(F.softplus(log_real_z, 1)))

            ''' ############## Discriminator の unsupervised dataを学習 [fake data]##################### '''

            noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
            fake_data = self.netG(noise)
            # label.fill_(fake_label)
            outputD_unlabel_fake, _ = self.netD(fake_data.detach())

            # log_fake_z = self.log_sum_exp(outputD_unlabel_fake, 1)
            log_fake_z = torch.logsumexp(outputD_unlabel_fake, 1)
            log_fake_d = F.softplus(log_fake_z, 1)
            errD_unlabel_fake = torch.mean(log_fake_d, 0)

            errD = errD_label + (errD_unlabel_real + errD_unlabel_fake)
            # errD = errD_label
            errD.backward()
            self.optimizerD.step()

            '''################# Generator の fake dataを学習 #####################'''
            self.netG.zero_grad()
            # label.fill_(real_label)
            # _, feature_unlabel_real = self.netD(unlabel_data.detach(), self.feature_matching)
            outputG, feature_unlabel_fake = self.netD(fake_data, self.feature_matching)

            # if self.feature_matching is True:

            feature_unlabel_real = torch.mean(feature_unlabel_real, 0)
            feature_unlabel_fake = torch.mean(feature_unlabel_fake, 0)
            errG = self.criterionG(feature_unlabel_real.detach(), feature_unlabel_fake)
            # else:

                # errG = self.criterionG(outputG, label)

            errG.backward()
            D_G_z2 = outputG.mean().item()
            self.optimizerG.step()


            self.logger.progress(iter_id)

            if iter_id % self.log_interval  == 0:

                _, predicted = torch.max(outputD_label, 1)
                correct = (predicted == label_target).sum().item()
                total = label_target.size(0)
                loss_acc = errD_label.item()
                self.logger.train_log(loss_acc, correct, total, iter_id)

                self.test(test_loader, iter_id)
                self.netD.train()

            #     # ロスを保存
            # self.G_losses.append(errG.item())
            # self.D_losses.append(errD.item())

            if (iter_id % 500 == 0) or (iter_id == self.iters):
                with torch.no_grad():
                    fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                self.save(self.img_list, 'img')
                self.save(self.G_losses, 'errorG')
                self.save(self.D_losses, 'errorD')


    def test(self, test_loader, iter_id):

        self.netD.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.netD(data)
                test_loss += self.criterionD(output, target).item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        log = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, total, 100. * correct / total)

        # self.logger.write(log)
        self.logger.valid_log(test_loss, correct, total, iter_id)


    def save(self, file, file_type):

        path = os.path.join(self.save_dir, file_type + '_' + self.prefix)
        if file_type is 'img':
            # image
            save_file = np.array([np.transpose(i.numpy(), (1, 2, 0)) for i in file])
        else:
            # error log
            save_file = np.array(file)

        np.save(path, save_file)



    def checkpoint(self, epoch):

        filename = os.path.join(self.out_dir, self.prefix)
        torch.save({'epoch': epoch + 1, 'logger': self.logger.state_dict()}, filename + '.iter')
        torch.save(self.net.state_dict(), filename + '.model')
        torch.save(self.optimizer.state_dict(), filename + '.state')

    def log_sum_exp(self, x, axis = 1):
        m = torch.max(x, dim = 1)[0]
        return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))