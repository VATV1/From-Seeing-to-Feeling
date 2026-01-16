import torch
from torchvision import models
from torch import nn
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from tqdm import tqdm
from utils import accuracy
import os
import wandb
try:
    from efficient_kan import KANLinear
except ImportError:
    raise ImportError("请先安装 efficient-kan 库: pip install efficient-kan")

def adapt_state_dict(state_dict):
    """
    Adapts the state dictionary's key names to match the expected keys of the ResNet model.
    """
    adapted_state_dict = {}
    for k, v in state_dict.items():
        # Remove the prefixed numbers from the key names
        new_key = '.'.join(k.split('.')[1:])
        adapted_state_dict[new_key] = v
    return adapted_state_dict


class KANAdaptiveFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, grid_size=5, spline_order=3):
        """
        KAN 自适应融合模块
        Args:
            input_dim: 单个模态的输入维度 (例如 512)
            output_dim: 融合后的输出维度 (可以保持 512，也可以增大或减小)
            grid_size: KAN 样条插值的网格大小 (默认 5，越大越精细但参数越多)
            spline_order: 样条函数的阶数 (默认 3)
        """
        super().__init__()
        # 拼接后的总输入维度
        total_input_dim = input_dim * 2


        self.kan_layer = KANLinear(
            total_input_dim,
            output_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )


        self.layer_norm = nn.LayerNorm(output_dim)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, rgb_features, tactile_features):
        """
        Args:
            rgb_features: [Batch, 512]
            tactile_features: [Batch, 512]
        Returns:
            fused_features: [Batch, output_dim]
        """
        combined = torch.cat((rgb_features, tactile_features), dim=1)

        fused = self.kan_layer(combined)

        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused

class LinearClassifier(nn.Module):
    def __init__(self, position_num_classes, softness_num_classes , checkpoint_path, nn_model='resnet18', pretrained=True):
        super(LinearClassifier, self).__init__()
        self.nn_model = nn_model
        self.rgb_encoder = self.create_resnet_encoder(3)
        self.tactile_encoder = self.create_resnet_encoder(3)
        self.fusion_module = KANAdaptiveFusion(input_dim=512, output_dim=512)
        if pretrained:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)

            # Adapt the state dictionary key names
            adapted_rgb_state_dict = adapt_state_dict(checkpoint['state_dict_vis'])
            adapted_tactile_state_dict = adapt_state_dict(checkpoint['state_dict_tac'])

            # Load the state dict for the visual and tactile encoders
            self.rgb_encoder.load_state_dict(adapted_rgb_state_dict, strict=False)
            self.tactile_encoder.load_state_dict(adapted_tactile_state_dict, strict=False)

            # Freeze the weights of the encoders
            for param in self.rgb_encoder.parameters():
                param.requires_grad = False
            for param in self.tactile_encoder.parameters():
                param.requires_grad = False
        # Assuming the output features of both encoders are of size 512 (e.g., for ResNet-18)
        # Adjust this if the size is different
        self.softness_layer = nn.Linear(512, softness_num_classes)
        self.position_layer = nn.Linear(512, position_num_classes)


    def create_resnet_encoder(self, n_channels):
        """Create a ResNet encoder based on the specified model type."""
        if self.nn_model == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif self.nn_model == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        if n_channels != 3:
            resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        features = list(resnet.children())[:-2]  # Exclude the avgpool and fc layers
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        features.append(nn.Flatten())
        return nn.Sequential(*features)

    def forward(self, rgb_input, tactile_input):
        rgb_features = self.rgb_encoder(rgb_input)
        tactile_features = self.tactile_encoder(tactile_input)


        # Concatenate the features from both encoders
        # combined_features = torch.cat((rgb_features, tactile_features), dim=1)
        # combined_features = self.fusion_module(rgb_features, tactile_features)
        return self.position_layer(tactile_features), self.softness_layer(tactile_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = '/media/lab404/dataset2/AVTNet/runs/Nov27_14-53-48_lab404/model_230_best_CAL.pth'
print(f"Using device: {device}")
linear_classifier = LinearClassifier(position_num_classes=2, softness_num_classes=5,  checkpoint_path=checkpoint_path, nn_model='resnet18', pretrained=True)
linear_classifier = linear_classifier.to(device)

batch_size = 32
num_workers = 16
use_wandb = False
dataset = ContrastiveLearningDataset(root_folder='/media/lab404/dataset2/AVTNet/DOVT-dataset')
train_dataset = dataset.get_dataset('calandra_label_train', 2)
test_dataset = dataset.get_dataset('calandra_label_test', 2,)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=False, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, drop_last=False, pin_memory=True)

optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
criterion = torch.nn.CrossEntropyLoss().to(device)

epochs = 100
if use_wandb:
    # init wandb
    wandb.init(project="calandra_object_wise_linear_classifier", name="resnet18_pretrained")
    if not os.path.exists(f"runs/{wandb.run.name}"):
        os.makedirs(f"runs/{wandb.run.name}")
    subfolder = wandb.run.name
else:
    subfolder = "test"

best_position_accuracy = 0
best_softness_accuracy = 0

for epoch in range(epochs):
    top1_position_train_accuracy = 0
    top1_softness_train_accuracy = 0
    epoch_loss = 0
    linear_classifier.train()
    pbar = tqdm(train_loader)
    for counter, data in enumerate(pbar):
        rgb_image_q, _, stacked_gelsight_images_q, _, position_label, softness_label  = data

        rgb_image_q = rgb_image_q.to(device)
        stacked_gelsight_images_q = stacked_gelsight_images_q.to(device)
        position_label = position_label.to(device)
        softness_label = softness_label.to(device)


        position_logits, softness_logits = linear_classifier(rgb_image_q, stacked_gelsight_images_q)
        loss_position = criterion(position_logits, position_label)
        loss_softness = criterion(softness_logits, softness_label)
        loss = loss_position + loss_softness
        epoch_loss += loss.item()

        top1_position = accuracy(position_logits, position_label, topk=(1,))
        top1_position_train_accuracy += top1_position[0]

        top1_softness = accuracy(softness_logits, softness_label, topk=(1,))
        top1_softness_train_accuracy += top1_softness[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the progress bar message
        pbar.set_description(
            f"Epoch {epoch}: Loss: {epoch_loss / (counter + 1):.2f}\tPosition Train Accuracy: {top1_position_train_accuracy.item() / (counter + 1):.2f}\tSoftness Train Accuracy: {top1_softness_train_accuracy.item() / (counter + 1):.2f}")
    scheduler.step()
    epoch_loss /= (len(train_loader))
    top1_position_train_accuracy /= (len(train_loader))
    top1_softness_train_accuracy /= (len(train_loader))

    # # save the model
    # if top1_train_accuracy > best_train_accuracy:
    #     torch.save(linear_classifier.state_dict(), f"runs/{subfolder}/linear_classifier_{epoch}_best_object_wise.pth")
    #     best_train_accuracy = top1_train_accuracy

    top1_position_accuracy = 0
    top1_softness_accuracy = 0

    linear_classifier.eval()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for counter, data in enumerate(pbar):
            rgb_image_q, _, stacked_gelsight_images_q, _, position_label, softness_label = data

            rgb_image_q = rgb_image_q.to(device)
            stacked_gelsight_images_q = stacked_gelsight_images_q.to(device)
            position_label = position_label.to(device)
            softness_label = softness_label.to(device)

            position_logits, softness_logits = linear_classifier(rgb_image_q, stacked_gelsight_images_q)

            top1_position = accuracy(position_logits, position_label, topk=(1,))
            top1_softness = accuracy(softness_logits, softness_label, topk=(1,))

            top1_position_accuracy += top1_position[0]
            top1_softness_accuracy += top1_softness[0]

            # update the progress bar message
            pbar.set_description(
                f"Epoch {epoch}:\tTrain Position Accuracy: {top1_position_train_accuracy.item():.2f}\tTrain Softness Accuracy: {top1_softness_train_accuracy.item():.2f}"
                f"\tTest Position Accuracy: {top1_position_accuracy.item() / (counter + 1):.2f}\tTest Softness Accuracy: {top1_softness_accuracy.item() / (counter + 1):.2f}"
                )

    top1_position_accuracy /= (len(test_loader))

    top1_softness_accuracy /= (len(test_loader))

    # save the model
    if top1_position_accuracy > best_position_accuracy and top1_softness_accuracy > best_softness_accuracy:
        torch.save(linear_classifier.state_dict(), f"runs/{subfolder}/linear_classifier_{epoch}_best_object_wise.pth")
        best_position_accuracy = top1_position_accuracy
        best_softness_accuracy = top1_softness_accuracy


    if use_wandb:
        wandb.log({"train_position_accuracy": top1_position_train_accuracy,
                   "train_softness_accuracy": top1_softness_train_accuracy,
                   "test_position_accuracy": top1_position_accuracy,
                   "test_softness_accuracy": top1_softness_accuracy,
                   "epoch_loss": epoch_loss})
    print(
        f"Epoch {epoch}:\tEpoch Loss: {epoch_loss:.2f}\tTrain Position Accuracy: {top1_position_train_accuracy.item():.2f}\tTrain Softness Accuracy: {top1_softness_train_accuracy.item():.2f}"
        f"\tTest Position Accuracy: {top1_position_accuracy.item():.2f}\tTest Softness Accuracy: {top1_softness_accuracy.item():.2f}")

