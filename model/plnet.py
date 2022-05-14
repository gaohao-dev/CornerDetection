import torch.nn as nn
import torch
# from torchvision.models import inception_v3
from torchvision.models import resnet50
import torch.nn.functional as F
import sys

sys.path.insert(0, "/home/haogao/works/PLN/")
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

class Inceptionbase(nn.Module):
    def __init__(self):
        super(Inceptionbase, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.backbone.eval()

    def forward(self, x):
        # x = self.backbone.Conv2d_1a_3x3.forward(x)
        # x = self.backbone.Conv2d_2a_3x3.forward(x)
        # x = self.backbone.Conv2d_2b_3x3.forward(x)
        # x = self.backbone.maxpool1.forward(x)
        # x = self.backbone.Conv2d_3b_1x1.forward(x)
        # x = self.backbone.Conv2d_4a_3x3.forward(x)
        # x = self.backbone.maxpool2.forward(x)
        # x = self.backbone.Mixed_5b.forward(x)
        # x = self.backbone.Mixed_5c.forward(x)
        # x = self.backbone.Mixed_5d.forward(x)
        # x = self.backbone.Mixed_6a.forward(x)
        # x = self.backbone.Mixed_6b.forward(x)
        # x = self.backbone.Mixed_6c.forward(x)
        # x = self.backbone.Mixed_6d.forward(x)
        # x = self.backbone.Mixed_6e.forward(x)
        # x = self.backbone.Mixed_7a.forward(x)
        # x = self.backbone.Mixed_7b.forward(x)
        # feat = self.backbone.Mixed_7c.forward(x)
        x=self.backbone.conv1(x)
        x=self.backbone.bn1(x)
        x=self.backbone.relu(x)
        x=self.backbone.maxpool(x)

        x=self.backbone.layer1(x)
        x=self.backbone.layer2(x)
        x=self.backbone.layer3(x)
        feat=self.backbone.layer4(x)

        return feat


class PLNcommon(nn.Module):
    def __init__(self):
        super(PLNcommon, self).__init__()
        self.conv = nn.Sequential(
            # 2048是backbone网络最终输出的通道数
            nn.Conv2d(in_channels=2048, out_channels=1536, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            nn.Conv2d(in_channels=1536, out_channels=1536, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            nn.Conv2d(in_channels=1536, out_channels=1536, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PLNpred(nn.Module):
    def __init__(self, args):
        super(PLNpred, self).__init__()
        channels = 2 * args.B * (1 + 20 + 2 + 2 * args.S)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=1536, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            nn.Conv2d(in_channels=1536, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.ASPP=ASPP(in_channels=channels,output_stride=8)
        # self.dilation = nn.Sequential(
        #     nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=2,
        #               dilation=(2, 2)),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=2,
        #               dilation=(2, 2)),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=4,
        #               dilation=(4, 4)),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=8,
        #               dilation=(8, 8)),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=16,
        #               dilation=(16, 16)),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               dilation=(1, 1)),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               dilation=(1, 1)),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU()
        # )

    def forward(self, x):
        x = self.conv(x)
        x=self.ASPP(x)
        # print(x.size())
        # x = self.dilation(x)

        return x

# 借鉴deeplab的网络结构，把原网络的串联的空间金字塔结构换为ASPP结构
def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True))

class ASPP(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASPP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, 204, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 204, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 204, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 204, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 204, 1, bias=False),
            nn.BatchNorm2d(204),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(204 * 5, 204, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(204)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
        return x


class PLNet(nn.Module):
    def __init__(self, args):
        super(PLNet, self).__init__()
        self.args = args
        self.n_classes = args.n_classes
        self.basenetwork = Inceptionbase()
        self.plncommon = PLNcommon()
        self.plnlt = PLNpred(args)
        self.plnrt = PLNpred(args)
        self.plnlb = PLNpred(args)
        self.plnrb = PLNpred(args)

    def forward(self, imgs):
        features = self.basenetwork(imgs)
        common = self.plncommon(features)
        lt = self.plnlt(common)
        rt = self.plnrt(common)
        lb = self.plnlb(common)
        rb = self.plnrb(common)
        lt=lt.permute(0,2,3,1)
        rt=rt.permute(0,2,3,1)
        lb=lb.permute(0,2,3,1)
        rb=rb.permute(0,2,3,1)
        lt = lt.view(self.args.batch_size, self.args.S * self.args.S, self.args.B * 2, -1)
        rt = rt.view(self.args.batch_size, self.args.S * self.args.S, self.args.B * 2, -1)
        lb = lb.view(self.args.batch_size, self.args.S * self.args.S, self.args.B * 2, -1)
        rb = rb.view(self.args.batch_size, self.args.S * self.args.S, self.args.B * 2, -1)
        nlt, nrt, nlb, nrb = self.normalize(lt, rt, lb, rb)
        return nlt, nrt, nlb, nrb

    def normalize(self, lt, rt, lb, rb):
        S = self.args.S
        m = 1 + 20 + 2 + 2 * S
        softmax = nn.Softmax(dim=3)
        nlt, nrt, nlb, nrb = lt.clone(), rt.clone(), lb.clone(), rb.clone()
        # 处理lt预测值
        nlt[:, :, :, 3:23] = softmax(lt[:, :, :, 3:23])
        nlt[:, :, :, 23:23 + S] = softmax(lt[:, :, :, 23:23 + S])
        nlt[:, :, :, 23 + S:23 + 2 * S] = softmax(lt[:, :, :, 23 + S:23 + 2 * S])

        # 处理rt预测值
        nrt[:, :, :, 3:23] = softmax(rt[:, :, :, 3:23])
        nrt[:, :, :, 23:23 + S] = softmax(rt[:, :, :, 23:23 + S])
        nrt[:, :, :, 23 + S:23 + 2 * S] = softmax(rt[:, :, :, 23 + S:23 + 2 * S])
        # 处理lb预测值
        nlb[:, :, :, 3:23] = softmax(lb[:, :, :, 3:23])
        nlb[:, :, :, 23:23 + S] = softmax(lb[:, :, :, 23:23 + S])
        nlb[:, :, :, 23 + S:23 + 2 * S] = softmax(lb[:, :, :, 23 + S:23 + 2 * S])
        # 处理rb预测值
        nrb[:, :, :, 3:23] = softmax(rb[:, :, :, 3:23])
        nrb[:, :, :, 23:23 + S] = softmax(rb[:, :, :, 23:23 + S])
        nrb[:, :, :, 23 + S:23 + 2 * S] = softmax(rb[:, :, :, 23 + S:23 + 2 * S])

        return nlt, nrt, nlb, nrb

    # def detect_objects(self, predicted_locs, predicted_labels, predicted_scores, max_overlap, top_k):
    #     """
    #     Decipher the S*S*B*4 locations and class scores (output of the tiny_detector) to detect objects.
    #
    #     For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
    #
    #     :param predicted_locs: predicted locations/boxes , a tensor of dimensions tensor([batch_size,S * S * B * 4,4])
    #     :param predicted_labels:
    #     :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 441, 1)
    #     :param min_score: minimum threshold for a box to be considered a match for a certain class
    #     :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    #     :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    #     :return: :return: detections (boxes, labels, and scores), lists of length batch_size
    #     """
    #     batch_size = predicted_locs.size(0)
    #     # n_priors是生成的boxs，m*4的向量
    #     # n_priors = self.priors_cxcy.size(0)
    #     # # 这个归一化在预测时进行和预测完进行区别大不大？
    #     # predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 441, n_classes)
    #     # N个类别进行回归，处理不同，他们的网络直接预测的是
    #     # Lists to store final predicted boxes, labels, and scores for all images in batch
    #     all_images_boxes = list()
    #     all_images_labels = list()
    #     all_images_scores = list()
    #
    #     assert predicted_locs.size(1) == predicted_scores.size(1) == predicted_labels.size(1)
    #
    #     for i in range(batch_size):
    #         # Decode object coordinates from the form we regressed predicted boxes to
    #         # 最终的到一个盒子即可，我们不需要提前创建priors_box了
    #         decoded_locs = predicted_locs[i]
    #
    #         image_boxes = list()
    #         image_labels = list()
    #         image_scores = list()
    #         # 返回分数以及对应的label值（441*n_classes）->(1*441),441个值以及label
    #         # max_scores, best_label = predicted_scores[i].max(dim=1)  # (441)
    #
    #         # Check for each class
    #         for c in range(1, self.n_classes):
    #             # Keep only predicted boxes and scores where scores for this class are above the minimum score
    #             # 只有分数足够大才进行保存
    #
    #             # class_scores = predicted_scores[i][:, c]  # (441)
    #             class_scores = predicted_scores[i]
    #             class_st = (predicted_labels[i][:, 0] == c)
    #
    #             # 符合条件的框框的个数
    #             n_class_st = class_st.sum().item()
    #             if n_class_st == 0:
    #                 continue
    #             # 重新生成盒子，只保留符合条件的盒子
    #             class_scores = class_scores[class_st]  # (n_qualified), n_min_score <= 441
    #             class_decoded_locs = decoded_locs[class_st]  # (n_qualified, 4)
    #
    #             # Sort predicted boxes and scores by scores
    #             # 进行了排序的呀！！！
    #             # 他的思路是每个类都可以对20个盒子进行排序，所以可能出现20*441个数量的盒子（过多了），我们的代码思路是，对每个盒子进行存储
    #             class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
    #             class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)
    #
    #             # Find the overlap between predicted boxes
    #             # 排序之后的盒子，然后进行调用，进行计算重合度,传入的参数值相等
    #             overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)
    #
    #             # Non-Maximum Suppression (NMS)
    #
    #             # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
    #             # 1 implies suppress, 0 implies don't suppress
    #             suppress = torch.zeros(n_above_min_score, dtype=torch.uint8).to(device)  # (n_qualified)
    #
    #             # Consider each box in order of decreasing scores
    #             for box in range(class_decoded_locs.size(0)):
    #                 # If this box is already marked for suppression
    #                 if suppress[box] == 1:
    #                     continue
    #
    #                 # Suppress boxes whose overlaps (with current box) are greater than maximum overlap
    #                 # Find such boxes and update suppress indices
    #                 suppress = torch.max(suppress, (overlap[box] > max_overlap).to(torch.uint8))
    #                 # The max operation retains previously suppressed boxes, like an 'OR' operation
    #
    #                 # Don't suppress this box, even though it has an overlap of 1 with itself
    #                 suppress[box] = 0
    #             # 上述操作是把被抑制的标签赋值为1，保留下来的仍为0
    #             # Store only unsuppressed boxes for this class
    #             image_boxes.append(class_decoded_locs[1 - suppress])
    #             image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
    #             image_scores.append(class_scores[1 - suppress])
    #
    #         # If no object in any class is found, store a placeholder for 'background'
    #         if len(image_boxes) == 0:
    #             image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
    #             image_labels.append(torch.LongTensor([0]).to(device))
    #             image_scores.append(torch.FloatTensor([0.]).to(device))
    #
    #         # Concatenate into single tensors
    #         image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
    #         image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
    #         image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
    #         n_objects = image_scores.size(0)
    #
    #         # Keep only the top k objects
    #         if n_objects > top_k:
    #             image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
    #             image_scores = image_scores[:top_k]  # (top_k)
    #             image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
    #             image_labels = image_labels[sort_ind][:top_k]  # (top_k)
    #
    #         # Append to lists that store predicted boxes and scores for all images
    #         all_images_boxes.append(image_boxes)
    #         all_images_labels.append(image_labels)
    #         all_images_scores.append(image_scores)
    #
    #     return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

    def inference(self, pred_lt, pred_rt, pred_lb, pred_rb):
        """
        :param pred_lt:
        :param pred_rt:
        :param pred_lb:
        :param pred_rb:
        :return: predicted_locs,predicted_labels,predicted_scores:tensor([batch_size,S * S * B * 8,4])
        tensor([batch_size,S * S * B * 8,1]),tensor([batch_size,S * S * B * 8,1])
        """
        pred_lt_p, pred_lt_xy, pred_lt_q, pred_lt_lx, pred_lt_ly = decompose(pred_lt, self.args.S)
        pred_rt_p, pred_rt_xy, pred_rt_q, pred_rt_lx, pred_rt_ly = decompose(pred_rt, self.args.S)
        pred_lb_p, pred_lb_xy, pred_lb_q, pred_lb_lx, pred_lb_ly = decompose(pred_lb, self.args.S)
        pred_rb_p, pred_rb_xy, pred_rb_q, pred_rb_lx, pred_rb_ly = decompose(pred_rb, self.args.S)

        batch_size = self.args.batch_size
        S = self.args.S
        B = self.args.B
        predicted_locs = torch.zeros([batch_size, S * S * B * 4, 4])
        predicted_labels = torch.zeros([batch_size, S * S * B * 4, 1])
        predicted_scores = torch.zeros([batch_size, S * S * B * 4, 1])
        lt_boxs = [[] for i in range(self.args.batch_size)]
        rt_boxs = [[] for i in range(self.args.batch_size)]
        lb_boxs = [[] for i in range(self.args.batch_size)]
        rb_boxs = [[] for i in range(self.args.batch_size)]
        batch_size = self.args.batch_size
        S = self.args.S
        B = self.args.B
        for bt in range(batch_size):
            for cenx in range(S):
                for ceny in range(S):
                    for j in range(B):
                        possibility_lt, possibility_rt, possibility_lb, possibility_rb = 0, 0, 0, 0
                        temp_lt_box = []
                        temp_rt_box = []
                        temp_lb_box = []
                        temp_rb_box = []
                        for corx in range(cenx + 1):
                            for cory in range(ceny + 1):
                                for cenlx in range(S):
                                    for cenly in range(S):
                                        for corlx in range(S):
                                            for corly in range(S):
                                                for n in range(S):
                                                    temp_lt = pred_lt_p[bt, cenx * S + ceny, j, 0] * pred_lt_p[
                                                        bt, corx * S + cory, j + B, 0] * pred_lt_q[
                                                                  bt, cenx * S + ceny, j, n] * pred_lt_q[
                                                                  bt, corx * S + cory, j + B, n] * (
                                                                      pred_lt_lx[bt, cenx * S + ceny, j, cenlx] *
                                                                      pred_lt_ly[bt, cenx * S + ceny, j, cenly] +
                                                                      pred_lt_lx[bt, corx * S + cory, j + B, corlx] *
                                                                      pred_lt_ly[bt, corx * S + cory, j + B, corly]) / 2
                                                    if (temp_lt > possibility_lt):
                                                        temp_lt_box = [temp_lt, cenx, ceny, j, corx, cory, n]
                        lt_boxs[bt].append(temp_lt_box)
                        for corx in range(cenx + 1):
                            for cory in range(ceny, S):
                                for cenlx in range(S):
                                    for cenly in range(S):
                                        for corlx in range(S):
                                            for corly in range(S):
                                                for n in range(S):
                                                    temp_rt = pred_rt_p[bt, cenx * S + ceny, j, 0] * pred_rt_p[
                                                        bt, corx * S + cory, j + B, 0] * pred_rt_q[
                                                                  bt, cenx * S + ceny, j, n] * pred_rt_q[
                                                                  bt, corx * S + cory, j + B, n] * (
                                                                      pred_rt_lx[bt, cenx * S + ceny, j, cenlx] *
                                                                      pred_rt_ly[bt, cenx * S + ceny, j, cenly] +
                                                                      pred_rt_lx[bt, corx * S + cory, j + B, corlx] *
                                                                      pred_rt_ly[bt, corx * S + cory, j + B, corly]) / 2
                                                    if (temp_rt > possibility_rt):
                                                        temp_rt_box = [temp_rt, cenx, ceny, j, corx, cory, n]
                        rt_boxs[bt].append(temp_rt_box)

                        for corx in range(cenx, S):
                            for cory in range(ceny + 1):
                                for cenlx in range(S):
                                    for cenly in range(S):
                                        for corlx in range(S):
                                            for corly in range(S):
                                                for n in range(S):
                                                    temp_lb = pred_lb_p[bt, cenx * S + ceny, j, 0] * pred_lb_p[
                                                        bt, corx * S + cory, j + B, 0] * pred_lb_q[
                                                                  bt, cenx * S + ceny, j, n] * pred_lb_q[
                                                                  bt, corx * S + cory, j + B, n] * (
                                                                      pred_lb_lx[bt, cenx * S + ceny, j, cenlx] *
                                                                      pred_lb_ly[bt, cenx * S + ceny, j, cenly] +
                                                                      pred_lb_lx[bt, corx * S + cory, j + B, corlx] *
                                                                      pred_lb_ly[bt, corx * S + cory, j + B, corly]) / 2
                                                    if (temp_lb > possibility_lb):
                                                        temp_lb_box = [temp_lb, cenx, ceny, j, corx, cory, n]
                        lb_boxs[bt].append(temp_lb_box)

                        for corx in range(cenx, S):
                            for cory in range(ceny, S):
                                for cenlx in range(S):
                                    for cenly in range(S):
                                        for corlx in range(S):
                                            for corly in range(S):
                                                for n in range(S):
                                                    temp_rb = pred_rb_p[bt, cenx * S + ceny, j, 0] * pred_rb_p[
                                                        bt, corx * S + cory, j + B, 0] * pred_rb_q[
                                                                  bt, cenx * S + ceny, j, n] * pred_rb_q[
                                                                  bt, corx * S + cory, j + B, n] * (
                                                                      pred_rb_lx[bt, cenx * S + ceny, j, cenlx] *
                                                                      pred_rb_ly[bt, cenx * S + ceny, j, cenly] +
                                                                      pred_rb_lx[bt, corx * S + cory, j + B, corlx] *
                                                                      pred_rb_ly[bt, corx * S + cory, j + B, corly]) / 2
                                                    if (temp_rb > possibility_rb):
                                                        temp_rb_box = [temp_rb, cenx, ceny, j, corx, cory, n]
                        rb_boxs[bt].append(temp_rb_box)
        predicted_locs[:, 0:S * S * 2, :], predicted_labels[:, 0:S * S * 2, :], predicted_scores[:, 0:S * S * 2,
                                                                                :] = generate_box(self.args,
                                                                                                  pred_lt_xy,
                                                                                                  lt_boxs,
                                                                                                  type='lt')
        predicted_locs[:, S * S * 2:S * S * 4, :], predicted_labels[:, S * S * 2:S * S * 4, :], predicted_scores[:,
                                                                                                0:S * S * 2,
                                                                                                :] = generate_box(
            self.args,
            pred_rt_xy,
            rt_boxs,
            type='rt')
        predicted_locs[:, S * S * 4:S * S * 6, :], predicted_labels[:, S * S * 4:S * S * 6, :], predicted_scores[:,
                                                                                                0:S * S * 2,
                                                                                                :] = generate_box(
            self.args,
            pred_lb_xy,
            lb_boxs,
            type='lb')
        predicted_locs[:, S * S * 6:S * S * 8, :], predicted_labels[:, S * S * 6:S * S * 8, :], predicted_scores[:,
                                                                                                0:S * S * 2,
                                                                                                :] = generate_box(
            self.args,
            pred_rb_xy,
            rb_boxs,
            type='rb')

        return predicted_locs, predicted_labels, predicted_scores


class MultiBoxLoss(nn.Module):
    def __init__(self, args):
        super(MultiBoxLoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.batch_size = args.batch_size
        self.S = args.S
        self.B = args.B
        self.w_class=args.loss_weights[0]
        self.w_coord=args.loss_weights[1]
        self.w_link=args.loss_weights[2]

    def forward(self, pred_lt, pred_rt, pred_lb, pred_rb, true_lt, true_rt, true_lb, true_rb):
        pred_lt_p, pred_lt_xy, pred_lt_q, pred_lt_lx, pred_lt_ly = decompose(pred_lt, self.S)
        pred_rt_p, pred_rt_xy, pred_rt_q, pred_rt_lx, pred_rt_ly = decompose(pred_rt, self.S)
        pred_lb_p, pred_lb_xy, pred_lb_q, pred_lb_lx, pred_lb_ly = decompose(pred_lb, self.S)
        pred_rb_p, pred_rb_xy, pred_rb_q, pred_rb_lx, pred_rb_ly = decompose(pred_rb, self.S)
        true_lt_p, true_lt_xy, true_lt_q, true_lt_lx, true_lt_ly = decompose(true_lt, self.S)
        true_rt_p, true_rt_xy, true_rt_q, true_rt_lx, true_rt_ly = decompose(true_rt, self.S)
        true_lb_p, true_lb_xy, true_lb_q, true_lb_lx, true_lb_ly = decompose(true_lb, self.S)
        true_rb_p, true_rb_xy, true_rb_q, true_rb_lx, true_rb_ly = decompose(true_rb, self.S)

        # print("----------------标签：--------------------")
        # print("p:", true_lt_p[1, 0:5, 2, :])
        # print("xy:", true_lt_xy[1, 0:5, 2, :])
        # print("q:", true_lt_q[1, 0:5, 2, :])
        # print("lx:", true_lt_lx[1, 0:5, 2, :])
        # print("ly:", true_lt_ly[1, 0:5, 2, :])
        # print("----------------预测：--------------------")
        # print("p:", pred_lt_p[1, 0:5, 2, :])
        # print("xy:", pred_lt_xy[1, 0:5, 2, :])
        # print("q:", pred_lt_q[1, 0:5, 2, :])
        # print("lx:", pred_lt_lx[1, 0:5, 2, :])
        # print("ly:", pred_lt_ly[1, 0:5, 2, :])

        batch_size = self.batch_size
        losses_lt = 0
        losses_rt = 0
        losses_lb = 0
        losses_rb = 0
        for b in range(batch_size):
            losses_lt = losses_lt + self.batch_loss(pred_lt_p[b], pred_lt_xy[b], pred_lt_q[b], pred_lt_lx[b],
                                                    pred_lt_ly[b], true_lt_p[b],
                                                    true_lt_xy[b], true_lt_q[b], true_lt_lx[b], true_lt_ly[b])
            losses_rt = losses_rt + self.batch_loss(pred_rt_p[b], pred_rt_xy[b], pred_rt_q[b], pred_rt_lx[b],
                                                    pred_rt_ly[b], true_rt_p[b],
                                                    true_rt_xy[b], true_rt_q[b], true_rt_lx[b], true_rt_ly[b])
            losses_lb = losses_lb + self.batch_loss(pred_lb_p[b], pred_lb_xy[b], pred_lb_q[b], pred_lb_lx[b],
                                                    pred_lb_ly[b], true_lb_p[b],
                                                    true_lb_xy[b], true_lb_q[b], true_lb_lx[b], true_lb_ly[b])
            losses_rb = losses_rb + self.batch_loss(pred_rb_p[b], pred_rb_xy[b], pred_rb_q[b], pred_rb_lx[b],
                                                    pred_rb_ly[b], true_rb_p[b],
                                                    true_rb_xy[b], true_rb_q[b], true_rb_lx[b], true_rb_ly[b])

        losses = (losses_lt + losses_lt + losses_lb + losses_rb)/batch_size

        return losses

    def batch_loss(self, pred_p, pred_xy, pred_q, pred_lx, pred_ly, true_p, true_xy, true_q,
                   true_lx, true_ly):
        S = self.S
        B = self.B
        batchloss = torch.zeros(1)
        batchloss=batchloss.to(device)
        pred_p=pred_p.to(device)
        pred_xy = pred_xy.to(device)
        pred_q = pred_q.to(device)
        pred_lx = pred_lx.to(device)
        pred_ly = pred_ly.to(device)
        true_p = true_p.to(device)
        true_xy = true_xy.to(device)
        true_q = true_q.to(device)
        true_lx = true_lx.to(device)
        true_ly = true_ly.to(device)
        # 计算损失
        for i in range(S * S):
            for j in range(B * 2):
                if int(true_p[i, j, 0].item()) == 1:
                    batchloss = batchloss + (pred_p[i, j, 0] - true_p[i, j, 0]) ** 2
                    batchloss = batchloss + ((pred_xy[i, j, 1:3] - true_xy[i, j, 1:3]) ** 2).sum()/2
                    batchloss = batchloss + ((pred_q[i, j, 3:23] - true_q[i, j, 3:23]) ** 2).sum()/20
                    batchloss = batchloss + ((pred_lx[i, j, 23:23 + S] - true_lx[i, j, 23:23 + S]) ** 2).sum()/S
                    batchloss = batchloss + (
                            (pred_ly[i, j, 23 + S:23 + 2 * S] - true_ly[i, j, 23 + S:23 + 2 * S]) ** 2).sum()/S
                else:
                    batchloss = batchloss + (pred_p[i, j, 0] - true_p[i, j, 0]) ** 2

        return batchloss
