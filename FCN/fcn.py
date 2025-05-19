import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from pathlib import Path


class FCNSegmenter(nn.Module):
    """
       全卷积神经网络（FCN）结构定义
       用于细胞图像的语义分割任务
       """
    def __init__(self):
        super(FCNSegmenter, self).__init__()
        # 编码器：三层卷积 + 池化
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 中间层：两个卷积提取高阶特征
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        # 解码器：反卷积恢复原图尺寸
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return torch.sigmoid(x)
    #  输出值范围 0~1，适合做二分类掩膜


class MyFCN:
    """
        FCN 分割模型封装类：
        - 包含模型初始化、加载数据、训练和预测保存
        - 接口结构模仿 myUnet，可直接替换使用
    """
    def __init__(self, img_shape=(512, 512), batch_size=2, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {self.device}")
        self.model = FCNSegmenter().to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def load_data(self, train_path="../npydata/imgs_train.npy",
                  mask_path="../npydata/imgs_mask_train.npy",
                  test_path="../npydata/imgs_test.npy"):
        """
                加载 .npy 格式的训练数据和测试数据
                输入 shape: [N, H, W, 1]，需转为 [N, 1, H, W]
        """
        imgs_train = np.load(train_path).astype("float32")
        masks_train = np.load(mask_path).astype("float32")
        imgs_test = np.load(test_path).astype("float32")

        # 转换为 PyTorch tensor，调整通道顺序为 [B, C, H, W]
        X = torch.tensor(imgs_train.transpose(0, 3, 1, 2))
        Y = torch.tensor(masks_train.transpose(0, 3, 1, 2))
        T = torch.tensor(imgs_test.transpose(0, 3, 1, 2))

        #强制将 labels 二值化
        Y = torch.tensor(masks_train.transpose(0, 3, 1, 2))
        Y = (Y > 0.5).float()  #  强制二值化为 0 或 1

        self.train_loader = DataLoader(TensorDataset(X, Y), batch_size=self.batch_size, shuffle=True)
        self.test_tensor = T.to(self.device)
        print(f"Loaded: {len(X)} train samples, {len(T)} test samples")

    def train(self, epochs=5, save_path="my_fcn_gpu.pth"):
        """
        训练模型并保存权重，同时输出每轮的 loss 和准确率
        """
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

                # 计算准确率（按像素分类）
                preds = (outputs > 0.5).float()
                acc = (preds == labels).float().mean().item()
                total_acc += acc
                count += 1

            avg_loss = total_loss / count
            avg_acc = total_acc / count
            print(f" Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")

    def predict_and_save(self, output_dir="results_fcn"):
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            preds = self.model(self.test_tensor)
            for i, pred in enumerate(preds):
                save_image(pred, os.path.join(output_dir, f"{i}.png"))
        print(f" Saved {len(preds)} prediction masks to {output_dir}")



if __name__ == "__main__":
    model = MyFCN()
    model.load_data(
        train_path="../npydata/imgs_train.npy",
        mask_path="../npydata/imgs_mask_train.npy",
        test_path="../npydata/imgs_test.npy"
    )
    model.train(epochs=10)
    model.predict_and_save("results_fcn")