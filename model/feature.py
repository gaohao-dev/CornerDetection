import torch
from torchvision.models import inception_v3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_feature(img):
    backbone = inception_v3(pretrained=True)
    backbone.aux_logits = False
    backbone.eval()
    backbone = backbone.to(device)
    x = backbone.Conv2d_1a_3x3.forward(img)
    x = backbone.Conv2d_2a_3x3.forward(x)
    x = backbone.Conv2d_2b_3x3.forward(x)
    x = backbone.maxpool1.forward(x)
    x = backbone.Conv2d_3b_1x1.forward(x)
    x = backbone.Conv2d_4a_3x3.forward(x)
    x = backbone.maxpool2.forward(x)
    x = backbone.Mixed_5b.forward(x)
    x = backbone.Mixed_5c.forward(x)
    x = backbone.Mixed_5d.forward(x)
    x = backbone.Mixed_6a.forward(x)
    x = backbone.Mixed_6b.forward(x)
    x = backbone.Mixed_6c.forward(x)
    x = backbone.Mixed_6d.forward(x)
    x = backbone.Mixed_6e.forward(x)
    x = backbone.Mixed_7a.forward(x)
    x = backbone.Mixed_7b.forward(x)
    feat = backbone.Mixed_7c.forward(x)
    return feat


if __name__ == '__main__':
    imgs = torch.zeros([2, 3, 448, 448])
    feats = extract_feature(imgs)
    print(feats.size())
    print("main函数！")
