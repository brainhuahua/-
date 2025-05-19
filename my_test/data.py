import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array


class myAugmentation(object):
    """
    一个用于图像增强的类：
    首先：分别读取训练图片和标签，然后将图片和标签合并用于下一步使用；
    然后：使用 Keras 的图像增强工具增强图像；
    最后：将增强后的图片分解，分别生成训练图片和标签图片。
    """
    def __init__(self, train_path="train", label_path="label", merge_path="merge",
                 aug_merge_path="aug_merge", aug_train_path="aug_train", aug_label_path="aug_label",
                 img_type="tif"):
        self.train_path = Path(train_path)
        self.label_path = Path(label_path)
        self.merge_path = Path(merge_path)
        self.aug_merge_path = Path(aug_merge_path)
        self.aug_train_path = Path(aug_train_path)
        self.aug_label_path = Path(aug_label_path)
        self.img_type = img_type

        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.train_imgs = sorted(self.train_path.glob(f"*.{img_type}"))
        self.label_imgs = sorted(self.label_path.glob(f"*.{img_type}"))
        self.slices = len(self.train_imgs)

        self.merge_path.mkdir(parents=True, exist_ok=True)
        self.aug_merge_path.mkdir(parents=True, exist_ok=True)

    def Augmentation(self):
        """
        开始图像增强操作：
        将训练图像和标签图像合并（mask 放入 B 通道），保存后进行增强
        """
        if len(self.train_imgs) != len(self.label_imgs) or len(self.train_imgs) == 0:
            print("训练图像和标签图像数量不一致或为空")
            return 0

        for i in range(len(self.train_imgs)):
            img_t = Image.open(self.train_imgs[i]).convert("RGB")
            img_l = Image.open(self.label_imgs[i]).convert("L")

            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)

            x_t[:, :, 2] = x_l[:, :, 0]  # 将 mask 合入 B 通道
            img_tmp = Image.fromarray(x_t.astype('uint8'))
            img_tmp.save(self.merge_path / f"{i}.{self.img_type}")

            x = x_t.reshape((1,) + x_t.shape)
            savedir = self.aug_merge_path / f"{i}"
            savedir.mkdir(parents=True, exist_ok=True)
            self.doAugmentate(x, savedir, str(i))

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):
        """
        使用 ImageDataGenerator 增强一张合并图像，生成 imgnum 张增强图像
        """
        i = 0
        for batch in self.datagen.flow(img,
                                       batch_size=batch_size,
                                       save_to_dir=save_to_dir,
                                       save_prefix=save_prefix,
                                       save_format=save_format):
            i += 1
            if i >= imgnum:
                break

    def splitMerge(self):
        """
        将增强后的 merge 图像再次拆分为训练图像和标签图像（分别保存）
        """
        for i in range(self.slices):
            path = self.aug_merge_path / str(i)
            train_imgs = sorted(path.glob(f"*.{self.img_type}"))

            path_train = self.aug_train_path / str(i)
            path_label = self.aug_label_path / str(i)
            path_train.mkdir(parents=True, exist_ok=True)
            path_label.mkdir(parents=True, exist_ok=True)

            for imgname in train_imgs:
                midname = imgname.stem
                img = cv2.imread(str(imgname))
                img_train = img[:, :, 2]
                img_label = img[:, :, 0]

                cv2.imwrite(str(path_train / f"{midname}_train.{self.img_type}"), img_train)
                cv2.imwrite(str(path_label / f"{midname}_label.{self.img_type}"), img_label)

    def splitTransform(self):
        """
        针对透视变换数据的切割操作，可自定义 merge 路径
        """
        path_merge = Path("deform/deform_norm2")
        path_train = Path("../deform/train")
        path_label = Path("deform/label")

        train_imgs = path_merge.glob(f"*.{self.img_type}")
        for imgname in train_imgs:
            midname = imgname.stem
            img = cv2.imread(str(imgname))
            img_train = img[:, :, 2]
            img_label = img[:, :, 0]

            cv2.imwrite(str(path_train / f"{midname}.{self.img_type}"), img_train)
            cv2.imwrite(str(path_label / f"{midname}.{self.img_type}"), img_label)


class dataProcess(object):
    """
    数据处理类：
    创建训练数据和测试数据（.npy 格式），并可加载使用。
    """
    def __init__(self, out_rows, out_cols,
                 data_path="../deform/train/train", label_path="../deform/train/label",
                 test_path="../test", npy_path="../npydata", img_type="tif"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = Path(data_path)
        self.label_path = Path(label_path)
        self.test_path = Path(test_path)
        self.npy_path = Path(npy_path)
        self.img_type = img_type
        self.npy_path.mkdir(parents=True, exist_ok=True)

    def create_train_data(self):
        print('Creating training images...')
        imgs = sorted(self.data_path.glob(f"*.{self.img_type}"))
        labels = sorted(self.label_path.glob(f"*.{self.img_type}"))

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for i, (img_p, label_p) in enumerate(zip(imgs, labels)):
            img = Image.open(img_p).convert("L").resize((self.out_cols, self.out_rows))
            label = Image.open(label_p).convert("L").resize((self.out_cols, self.out_rows))
            imgdatas[i] = img_to_array(img)
            imglabels[i] = img_to_array(label)

            if i % 100 == 0:
                print(f"Loaded {i}/{len(imgs)}")

        np.save(self.npy_path / 'imgs_train.npy', imgdatas)
        np.save(self.npy_path / 'imgs_mask_train.npy', imglabels)
        print("Train data saved to .npy files")

    def create_test_data(self):
        print('Creating test images...')
        imgs = sorted(self.test_path.glob(f"*.{self.img_type}"))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for i, img_p in enumerate(imgs):
            img = Image.open(img_p).convert("L").resize((self.out_cols, self.out_rows))
            imgdatas[i] = img_to_array(img)

        np.save(self.npy_path / 'imgs_test.npy', imgdatas)
        print("Test data saved.")

    def load_train_data(self):
        print("Loading train data...")
        imgs_train = np.load(self.npy_path / "imgs_train.npy").astype("float32") / 255.
        imgs_mask_train = np.load(self.npy_path / "imgs_mask_train.npy").astype("float32") / 255.

        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train = (imgs_mask_train > 0.5).astype("float32")
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print("Loading test data...")
        imgs_test = np.load(self.npy_path / "imgs_test.npy").astype("float32") / 255.
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test



if __name__ == "__main__":
# 以下注释掉的部分为数据增强代码，通过他们可以将数据进行增强

    #aug = myAugmentation()
    #aug.Augmentation()
    #aug.splitMerge()
    #aug.splitTransform()

    mydata = dataProcess(512,512)
    mydata.create_train_data()
    mydata.create_test_data()

    imgs_train,imgs_mask_train = mydata.load_train_data()
    print (imgs_train.shape,imgs_mask_train.shape)
