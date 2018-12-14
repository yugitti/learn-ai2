import torch
import os

class NNRun(object):

    def __init__(self, net, optimizer, criterion, logger, device, log_interval, out_dir, prefix):

        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.device = device
        self.log_interval = log_interval
        self.out_dir = out_dir
        self.prefix = '_' + prefix

    def train(self, train_loader, epoch):
        self.net.train()
        correct = 0
        total = 0
        loss_acc = 0
        for batch_id, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            loss_acc += loss.item()
            if batch_id % self.log_interval == 0:
                #             log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #                 epoch, batch_id * len(data), len(train_loader.dataset),
                #                 100. * batch_id / len(train_loader), loss.item())

                self.logger.progress(batch_id, len(train_loader), epoch)
        #             print(len(train_loader))
        #             logger.write(log)
        print(' ')
        self.logger.train_log(loss_acc, correct, total)
        #     log = 'EPOCH [{}]: Train Loss: {:.6f}'.format(epoch, loss.item())
        # log = 'EPOCH [{}]: Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #     epoch, loss, correct, total, 100. * correct / total)
        #     progress(batch_id, len(train_loader))
        #     print('\n')
        # self.logger.write(log)


    def test(self, test_loader):

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        log = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, total, 100. * correct / total)

        # self.logger.write(log)
        self.logger.valid_log(test_loss, correct, total)


    def checkpoint(self, epoch):

        filename = os.path.join(self.out_dir, self.prefix)
        torch.save({'epoch': epoch + 1, 'logger': self.logger.state_dict()}, filename + '.iter')
        torch.save(self.net.state_dict(), filename + '.model')
        torch.save(self.optimizer.state_dict(), filename + '.state')
