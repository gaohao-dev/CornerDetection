import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import tiny_detector, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

model=tiny_detector(20)
model.to(device)
model.eval()

def test(original_image, min_score, max_overlap, top_k):
    # Transform the image 对图片进行处理
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # 现在开始导入模型
    predicted_locs, predicted_scores, p, q, xy, linkx, linky = model(image.unsqueeze(0))
    print(p.size())
    print(q.size())
    print(xy.size())
    print(linkx.size())
    print(linky.size())
    predicted_locs,predicted_labels,predicted_scores=model.inference(p,q,xy,linkx,linky)
    print(predicted_locs.size())
    print(predicted_labels.size())
    print(predicted_scores.size())



if __name__ == '__main__':
    img_path = '/home/haogao/works/PLN/vocdata/VOC2007/JPEGImages/000005.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    test(original_image, min_score=0.2, max_overlap=0.5, top_k=200)

