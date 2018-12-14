import torch.nn as nn
import torch.nn.functional as F
import torch
import contextlib

def l2_normalize(d):

    # 画像の場合、batch x (WxHxC) x 1 x 1
    # この処理の意図は最終的にバッチごとにnormを計算するため
    d_reshape = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    #     d_norm = torch.norm(d_reshape, dim=1, keepdim=True) + 1e-8

    d_norm = torch.sqrt(torch.sum(d_reshape**2, dim=1, keepdim=True))

    return d / d_norm


def kl_div(log_probs, probs):

    #     qlogq = (probs * torch.log(probs)).sum()
    #     qlogp = (probs * log_probs).sum()
    #     return  (qlogq - qlogp).sum()

    return F.kl_div(log_probs, probs, reduction='sum')

#     return F.kl_div(log_probs, probs, reduction='elementwise_mean')

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


class VAT(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        super(VAT, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):

        # ラベル無しデータをネットワークに通し、predictを得る
        # VATのLossのbackpropagationでは、このmodel計算は含めないため
        # no_gradでwrapする
        with torch.no_grad():
            out = model(x)
            pred = F.softmax(out, dim=1)

        # 累積法を用いてVadvを計算する

        d = torch.rand(x.shape).sub(0.5).to(x.device)  ## ゼロを中心に乱数による初期値
        d = l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                adv_dist = kl_div(F.log_softmax(pred_hat, dim=1), pred)

                adv_dist.backward()
                d = l2_normalize(d.grad)
                model.zero_grad()

        ## LDSを計算する
        r_adv = d * self.eps
        pred_hat = model(x + r_adv)
        lds = kl_div(F.log_softmax(pred_hat, dim=1), pred)

        return lds
