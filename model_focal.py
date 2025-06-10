import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attn)

# Enhanced Encoder with Attention
class EnhancedEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(EnhancedEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.channel_attention = ChannelAttention(output_channels)
        self.spatial_attention = SpatialAttention()
        self.residual_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.spatial_attention(self.channel_attention(x))
        x += residual  # Add residual connection
        return x

# Multi-Scale Decoder with Skip Connections
class MultiScaleDecoder(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MultiScaleDecoder, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(input_channels, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self.align4 = nn.Conv2d(256, 512, kernel_size=1)
        self.align3 = nn.Conv2d(128, 256, kernel_size=1)
        self.align2 = nn.Conv2d(64, 128, kernel_size=1)
        self.align1 = nn.Conv2d(32, 64, kernel_size=1)

        self.fusion = nn.Conv2d(256 + 128 + 64 + 32, 256, kernel_size=1)  # Multi-scale feature fusion
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, enc1, enc2, enc3, enc4):
        x = F.relu(self.upconv1(enc4))
        x = F.relu(self.upconv2(x + F.interpolate(self.align3(enc3), size=x.shape[2:], mode="bilinear", align_corners=False)))
        x = F.relu(self.upconv3(x + F.interpolate(self.align2(enc2), size=x.shape[2:], mode="bilinear", align_corners=False)))
        x = F.relu(self.upconv4(x + F.interpolate(self.align1(enc1), size=x.shape[2:], mode="bilinear", align_corners=False)))

        # Multi-scale fusion
        fusion_features = torch.cat([x, 
                                      F.interpolate(enc4, size=x.shape[2:], mode="bilinear", align_corners=False),
                                      F.interpolate(enc3, size=x.shape[2:], mode="bilinear", align_corners=False),
                                      F.interpolate(enc2, size=x.shape[2:], mode="bilinear", align_corners=False)], dim=1)
        fused = F.relu(self.fusion(fusion_features))
        output = self.final_conv(fused)
        return output

class DynamicFeatureAggregation(nn.Module):
    def __init__(self, num_conditions, feature_dim):
        super(DynamicFeatureAggregation, self).__init__()
        self.num_conditions = num_conditions
        self.feature_dim = feature_dim
        self.attention_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim // 2, 1, kernel_size=1),
                nn.Sigmoid()
            )
            for _ in range(num_conditions)
        ])

    def forward(self, condition_features):
        weighted_features = []
        for i, feature in enumerate(condition_features):
            attention_weights = self.attention_networks[i](feature)
            weighted_features.append(feature * attention_weights)
        return sum(weighted_features)

class CrossConditionConsistencyLoss(nn.Module):
    def forward(self, feature1, feature2):
        return F.mse_loss(feature1, feature2)

class EnhancedMultiConditionFusionNet(nn.Module):
    def __init__(self, input_channels, num_conditions, num_classes):
        super(EnhancedMultiConditionFusionNet, self).__init__()
        self.num_conditions = num_conditions
        self.encoder1 = EnhancedEncoder(input_channels, 32)
        self.encoder2 = EnhancedEncoder(32, 64)
        self.encoder3 = EnhancedEncoder(64, 128)
        self.encoder4 = EnhancedEncoder(128, 256)
        
        self.dfa = DynamicFeatureAggregation(num_conditions, feature_dim=256)
        self.decoder = MultiScaleDecoder(input_channels=256, num_classes=num_classes)
        self.consistency_loss_fn = CrossConditionConsistencyLoss()

    def forward(self, x, x_condition=None):
        # Encode primary input
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        if x_condition is not None:
            # Encode condition input
            enc1_cond = self.encoder1(x_condition)
            enc2_cond = self.encoder2(enc1_cond)
            enc3_cond = self.encoder3(enc2_cond)
            enc4_cond = self.encoder4(enc3_cond)

            # Compute consistency loss
            pooled_enc4 = F.adaptive_avg_pool2d(enc4, (1, 1))
            pooled_enc4_cond = F.adaptive_avg_pool2d(enc4_cond, (1, 1))
            consistency_loss = self.consistency_loss_fn(pooled_enc4, pooled_enc4_cond)

            # Aggregate features across conditions
            enc4_fused = self.dfa([enc4, enc4_cond])
        else:
            consistency_loss = None
            enc4_fused = enc4

        # Decode fused features
        output = self.decoder(enc1, enc2, enc3, enc4_fused)
        return output, consistency_loss

class DynamicFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=14):
        super(DynamicFocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits, targets, class_weights):
        """
        Args:
            logits: Predicted logits [batch_size, num_classes, height, width]
            targets: Ground truth labels [batch_size, height, width]
            class_difficulty: Per-class difficulty weight [num_classes]
        """
        class_weights = class_weights.cuda()
        probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        probs = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # Flatten spatial dimensions
        targets = targets.view(-1)  # Flatten targets

        mask = targets >= 0  # Ignore invalid labels (-1)
        probs = probs[mask]
        targets = targets[mask]

        class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()  # Gather probabilities of true classes

        # Compute focal scaling factor
        focal_weight = (1 - class_probs) ** self.gamma

        # Apply class difficulty weight
        class_weight = class_weights[targets]
        loss = -class_weight * focal_weight * torch.log(class_probs + 1e-6)

        return loss.mean()

class IoULoss(nn.Module):
    def forward(self, logits, targets, eps=1e-6):
        logits = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 3, 1, 2)
        intersection = (logits * targets_one_hot).sum(dim=(2, 3))
        union = logits.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection
        iou = (intersection + eps) / (union + eps)
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, num_classes=14):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.focal_loss = DynamicFocalLoss(gamma=gamma, num_classes=num_classes)
        self.iou_loss = IoULoss()

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        iou = self.iou_loss(logits, targets)
        return self.alpha * focal + (1 - self.alpha) * iou