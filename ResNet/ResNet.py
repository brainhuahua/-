import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.relu(out)


class ResNetSegmenter(nn.Module):
    def __init__(self):
        super(ResNetSegmenter, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2)
        )

        self.middle = nn.Sequential(
            ResidualBlock(256, 512)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(512, 256),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(256, 128),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(128, 64),

            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


class MyResNetSeg:
    def __init__(self, batch_size=2, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" 当前使用设备: {self.device}")
        self.model = ResNetSegmenter().to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def load_data(self, train_path="npydata/imgs_train.npy",
                  mask_path="npydata/imgs_mask_train.npy",
                  test_path="npydata/imgs_test.npy"):

        imgs_train = np.load(train_path).astype("float32")
        masks_train = np.load(mask_path).astype("float32")
        imgs_test = np.load(test_path).astype("float32")

        X = torch.tensor(imgs_train.transpose(0, 3, 1, 2))
        Y = torch.tensor(masks_train.transpose(0, 3, 1, 2))
        Y = (Y > 0.5).float()  # 二值化标签
        T = torch.tensor(imgs_test.transpose(0, 3, 1, 2))

        self.train_loader = DataLoader(TensorDataset(X, Y), batch_size=self.batch_size, shuffle=True)
        self.test_tensor = T.to(self.device)
        print(f" 已加载训练样本 {len(X)}，测试样本 {len(T)}")

    def train(self, epochs=5, save_path="my_resnet_seg.pth"):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            count = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                acc = (preds == labels).float().mean().item()
                total_acc += acc
                count += 1

            print(f" Epoch {epoch+1}/{epochs} - Loss: {total_loss/count:.4f} - Acc: {total_acc/count:.4f}")
        torch.save(self.model.state_dict(), save_path)
        print(f" 模型已保存至: {save_path}")

    def predict_and_save(self, output_dir="results_resnet"):
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            preds = self.model(self.test_tensor)
            for i, pred in enumerate(preds):
                save_image(pred, os.path.join(output_dir, f"{i}.png"))
        print(f" 已保存预测图像到: {output_dir}")


if __name__ == "__main__":
    model = MyResNetSeg()
    model.load_data(
        train_path="../npydata/imgs_train.npy",
        mask_path="../npydata/imgs_mask_train.npy",
        test_path="../npydata/imgs_test.npy"
    )
    model.train(epochs=10)
    model.predict_and_save("results_resnet")