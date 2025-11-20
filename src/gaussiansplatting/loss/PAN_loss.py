from loss.base_loss import base_Loss
import torch


class Lpan(base_Loss):
    def __init__(self, lambda_Lpan):
        super(Lpan, self).__init__()
        self.lambda_Lpan = lambda_Lpan

    def forward(self, pan_image, gt_pan_image):
        # compute the L2 loss between the PAN image and the ground truth PAN image
        l_pan = torch.mean((pan_image - gt_pan_image) ** 2)
        return l_pan


class Lgradient_pan(base_Loss):
    def __init__(self, lambda_lgradient_pan):
        super(Lgradient_pan, self).__init__()
        self.lambda_lgradient_pan = lambda_lgradient_pan

    def forward(self, pan_image, gt_pan_image):
        # Compute the gradient of the PAN image
        grad_pan = torch.gradient(pan_image, dim=(-2, -1))
        grad_gt_pan = torch.gradient(gt_pan_image, dim=(-2, -1))

        # Compute the L2 loss between the gradients
        lgradient_pan = torch.mean((grad_pan[0] - grad_gt_pan[0]) ** 2) + torch.mean(
            (grad_pan[1] - grad_gt_pan[1]) ** 2
        )
        return lgradient_pan


# if __name__ == "__main__":
#     width = 528
#     height = 528
#     MSI_chan = 3
#     PAN_chan = 1
#     x= torch.rand(( PAN_chan, height, width))
#     y = torch.rand(( PAN_chan, height, width))
#     pan_loss = Lpan(lambda_Lpan=1.0)
#     l_pan = pan_loss(x, y)
#     print("Lpan:", l_pan.item())

#     grad_pan= Lgradient_pan(lambda_lgradient_pan=1.0)
#     l_gradient_pan = grad_pan(x, y)
#     print("Lpan loss:", l_pan.item())
