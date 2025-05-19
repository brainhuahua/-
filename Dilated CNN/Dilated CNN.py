import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image


class DilatedCNN(nn.Module):
    """
    使用扩张卷积的图像分割网络：
    - 无池化，全卷积结构
    - 扩大感受野，提取多尺度上下文
    """
    def __init__(self):
        super(DilatedCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),      # 普通卷积
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=2, dilation=2),  # 扩张卷积
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=4, dilation=4),
            nn.ReLU()
        )

        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=4, dilation=4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


class MyDilatedSeg:
    """
    Dilated CNN 模型封装：
    - 自动使用 GPU
    - 支持 .npy 数据加载
    - 一键训练 & 保存预测图像
    """
    def __init__(self, batch_size=2, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" 当前使用设备: {self.device}")
        self.model = DilatedCNN().to(self.device)
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
        Y = (Y > 0.5).float()
        T = torch.tensor(imgs_test.transpose(0, 3, 1, 2))

        self.train_loader = DataLoader(TensorDataset(X, Y), batch_size=self.batch_size, shuffle=True)
        self.test_tensor = T.to(self.device)
        print(f" 已加载训练样本 {len(X)}，测试样本 {len(T)}")

    def train(self, epochs=5, save_path="my_dilated_seg.pth"):
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

    def predict_and_save(self, output_dir="results_dilated"):
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            preds = self.model(self.test_tensor)
            for i, pred in enumerate(preds):
                save_image(pred, os.path.join(output_dir, f"{i}.png"))
        print(f" 预测结果保存至: {output_dir}")


if __name__ == "__main__":
    model = MyDilatedSeg()
    model.load_data(
        train_path="../npydata/imgs_train.npy",
        mask_path="../npydata/imgs_mask_train.npy",
        test_path="../npydata/imgs_test.npy"
    )
    model.train(epochs=10)
    model.predict_and_save("results_dilated")
