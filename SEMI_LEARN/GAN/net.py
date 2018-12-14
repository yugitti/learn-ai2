import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        #         self.gpu = args.gpu

        ##upsampleinで得られる画像サイズ = (W - 1) x stride - 2xpadding + kernel + outputpadding

        self.model = nn.Sequential(
            # 乱数z, generatorのconv層への入力
            # 層の深さ ngfx8, kernel: 4, stride: 1, padding: 0
            # サイズ: (1-1)x 1 - 2x0 + 4 = 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 層の深さ ngfx4, kernel: 4, stride: 2, padding: 1
            # サイズ: (4-1)x 2 - 2x1 + 4 = 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 層の深さ ngfx2, kernel: 4, stride: 2, padding: 1
            # サイズ: (8-1)x 2 - 2x1 + 4 = 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 層の深さ ngfx2, kernel: 4, stride: 2, padding: 1
            # サイズ: (32-1)x 2 - 2x1 + 4 = 64
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()


            # # 層の深さ ngfx2, kernel: 4, stride: 2, padding: 1
            # # サイズ: (16-1)x 2 - 2x1 + 4 = 32
            # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # 層の深さ ngfx2, kernel: 4, stride: 2, padding: 1
            # # サイズ: (32-1)x 2 - 2x1 + 4 = 64
            # nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # nn.Tanh()
        )

        # self.model(weights_init)

    def forward(self, input):
        return self.model(input)


class MiniBatchDiscriminator(nn.Module):
    def __init__(self, A, B, C, device, batch_size):
        super(MiniBatchDiscriminator, self).__init__()
        self.A, self.B, self.C = A, B, C
        self.device = device
        self.eraser = torch.eye(batch_size).view(batch_size, 1, batch_size).to(self.device)
        # T_init = torch.randn([A, B, C])
        # self.T = nn.Parameter(T_init, requires_grad=True).to(device)
        T_init = torch.randn([A, B * C])
        self.T = nn.Parameter(T_init, requires_grad=True).to(device)

    def forward(self, x):
        # start = torch.cuda.Event(enable_timing=True)
        # interval1 = torch.cuda.Event(enable_timing=True)
        # interval2 = torch.cuda.Event(enable_timing=True)
        # interval3 = torch.cuda.Event(enable_timing=True)
        # interval4 = torch.cuda.Event(enable_timing=True)
        # interval4_1 = torch.cuda.Event(enable_timing=True)
        # interval5 = torch.cuda.Event(enable_timing=True)
        # interval6 = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        batch_size = x.size()[0]
        # self.T = (self.T).view([self.A, -1])
        m = x.mm(self.T)
        # interval1.record()

        m = m.view(-1, self.B, self.C)
        m = m.unsqueeze(-1)
        m_T = torch.transpose(m, 0, 3)
        # interval2.record()

        m = m.expand(batch_size, -1, -1, batch_size)
        m_T = m_T.expand(batch_size, -1, -1, batch_size)
        # interval3.record()

        norm2 = torch.sum(torch.abs(m - m_T), dim=2)
        # interval4.record()

        # eraser = torch.eye(batch_size).view(batch_size, 1, batch_size).to(self.device)
        eraser = self.eraser[:batch_size, :, :batch_size]
        # interval4_1.record()

        eraser = eraser.expand_as(norm2)

        # interval5.record()

        c_b2 = torch.exp(-(norm2 + 1e6 * eraser))
        o_b2 = torch.sum(c_b2, dim=2)
        # interval6.record()

        output = torch.cat((x, o_b2), 1)
        # end.record()

        # torch.cuda.synchronize()
        # print('interval[1]: {}'.format(start.elapsed_time(interval1)))
        # print('interval[2]: {}'.format(interval1.elapsed_time(interval2)))
        # print('interval[3]: {}'.format(interval2.elapsed_time(interval3)))
        # print('interval[4]: {}'.format(interval3.elapsed_time(interval4)))
        # print('interval[4_1]: {}'.format(interval4.elapsed_time(interval4_1)))
        # print('interval[5]: {}'.format(interval4_1.elapsed_time(interval5)))
        # print('interval[6]: {}'.format(interval5.elapsed_time(interval6)))
        # print('interval[7]: {}'.format(interval6.elapsed_time(end)))

        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, device, batch_size, minibatch=True):
        super(Discriminator, self).__init__()

        self.ndf = ndf
        self.A = ndf*8
        # self.A = ndf*8*4*4
        self.B = 128
        self.C = 16
        self.minibatch_flag = minibatch
        self.minibatch = MiniBatchDiscriminator(self.A, self.B, self.C, device, batch_size)

        self.model = nn.Sequential(
            # SIZE = (W + 2xpadding - kernel) / stride + 1
            # nc x 64 x 64 >> (64 + 2x1 - 4)/2 +1 = 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32 >> (32 + 2x1 - 4)/2 +1 = 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 16 x 16 >> (16 + 2x1 - 4)/2 +1 = 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 8 x 8 >> (8 + 2x1 - 4)/2 +1 = 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # # (ndf) x 4 x 4 >> (4 + 2x0 - 4)/1 +1 = 1 >> 1つの値を出力
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()

        )

        # self.fn = nn.Linear(ndf*8*4*4, 1)
        # self.fn1 = nn.Linear(ndf*8*4*4, ndf*8)
        self.fn1 = nn.Linear(ndf * 8 * 2 * 2, ndf * 8)
        self.fn2 = nn.Linear(ndf*8, 10)

        # self.fn_mb = nn.Linear(ndf*8*4*4 + self.B, 1)
        self.fn2_mb = nn.Linear(ndf*8 + self.B, 10)
        # self.sigmoid = nn.Sigmoid()

        # self.model(weights_init)

    def forward(self, h, feature_matching = False):
        # start = torch.cuda.Event(enable_timing=True)
        # interval1 = torch.cuda.Event(enable_timing=True)
        # interval2 = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        x = self.model(h)
        # x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = x.view(-1, self.ndf * 8 * 2 * 2)
        x = self.fn1(x)

        # interval1.record()

        if self.minibatch_flag  is True:
            x = self.minibatch(x)
            # interval2.record()
            output = self.fn2_mb(x)
            # end.record()
        else:
            output = self.fn2(x)
            # interval2.record()
        # output = self.sigmoid(output)

        # torch.cuda.synchronize()
        # print('interval[1]: {}'.format(start.elapsed_time(interval1)))
        # print('interval[2]: {}'.format(interval1.elapsed_time(interval2)))
        # print('interval[3]: {}'.format(interval2.elapsed_time(end)))

        return output, x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)