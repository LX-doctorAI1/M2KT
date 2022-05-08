import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .efficientnet_pytorch.model import EfficientNet

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
        )

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet121.classifier(out)
        return out


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.args = args
        print(f"=> creating model '{args.visual_extractor}'")
        if args.visual_extractor == 'densenet':
            self.model = DenseNet121(args.num_labels)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        elif args.visual_extractor == 'efficientnet':
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.num_labels)
        elif 'resnet' in args.visual_extractor:
            self.visual_extractor = args.visual_extractor
            self.pretrained = args.visual_extractor_pretrained
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            self.classifier = nn.Linear(2048, args.num_labels)
        else:
            raise NotImplementedError

        # load pretrained visual extractor
        if args.pretrain_cnn_file and args.pretrain_cnn_file != "":
            print(f'Load pretrained CNN model from: {args.pretrain_cnn_file}')
            checkpoint = torch.load(args.pretrain_cnn_file, map_location='cuda:{}'.format(args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f'Load pretrained CNN model from: official pretrained in ImageNet')

    def forward(self, images):
        if self.args.visual_extractor == 'densenet':
            patch_feats = self.model.densenet121.features(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))

            x = F.relu(patch_feats, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            labels = self.model.densenet121.classifier(x)

        elif self.args.visual_extractor == 'efficientnet':
            patch_feats = self.model.extract_features(images)
            # Pooling and final linear layer
            avg_feats = self.model._avg_pooling(patch_feats)
            x = avg_feats.flatten(start_dim=1)
            x = self.model._dropout(x)
            labels = self.model._fc(x)
        elif 'resnet' in self.visual_extractor:
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            labels = self.classifier(avg_feats)
        else:
            raise NotImplementedError

        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats, labels
