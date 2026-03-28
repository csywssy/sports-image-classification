import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================== 配置参数 ====================
RAW_DATA_DIR = r"D:/Thesis_new/dataset"  # 原始数据目录
TRAIN_CSV_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
TRAIN_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "train")  # 原始训练图片文件夹
TEST_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "test")    # 原始测试图片文件夹

OUTPUT_DIR = r"D:/Thesis_new/data"  # 整理后的数据输出目录
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "val")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "test")

VAL_RATIO = 0.2  # 验证集比例
RANDOM_SEED = 42

# ==================== 复制图片函数 ====================
def copy_images(df, source_dir, target_dir, is_test=False):
    copied_count = 0
    for idx, row in df.iterrows():
        img_name = str(row['image_ID']).strip()
        label = str(row['label']).strip() if not is_test else "unknown"

        src_path = os.path.join(source_dir, img_name)
        if not os.path.exists(src_path):
            # 尝试加上扩展名
            for ext in [".jpg", ".jpeg", ".png"]:
                if os.path.exists(src_path + ext):
                    src_path = src_path + ext
                    break
        if not os.path.exists(src_path):
            print(f"⚠ 找不到图片 {img_name}")
            continue

        dst_dir = os.path.join(target_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied_count += 1

    print(f"✅ 已复制 {copied_count} 张图片到 {target_dir}")
    return copied_count

# ==================== 主程序 ====================
def main():
    os.makedirs(TRAIN_OUTPUT, exist_ok=True)
    os.makedirs(VAL_OUTPUT, exist_ok=True)
    os.makedirs(TEST_OUTPUT, exist_ok=True)

    # 1. 读取训练 CSV
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"训练数据总数: {len(train_df)} 张图片，类别: {train_df['label'].nunique()}")

    # 2. 划分训练集和验证集
    train_data, val_data = train_test_split(
        train_df,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=train_df['label']
    )
    print(f"训练集: {len(train_data)} 张, 验证集: {len(val_data)} 张")

    # 3. 复制训练集和验证集
    copy_images(train_data, TRAIN_IMAGES_DIR, TRAIN_OUTPUT)
    copy_images(val_data, TRAIN_IMAGES_DIR, VAL_OUTPUT)

    # 4. 处理测试集
    if os.path.exists(TEST_CSV_PATH):
        test_df = pd.read_csv(TEST_CSV_PATH)
        copy_images(test_df, TEST_IMAGES_DIR, os.path.join(TEST_OUTPUT, "all_test_images"), is_test=True)
    else:
        print("⚠ 未找到测试集 CSV 文件，跳过测试集处理")

    print("\n🎯 数据集整理完成！文件夹结构如下：")
    print(f"  {OUTPUT_DIR}/train/")
    print(f"  {OUTPUT_DIR}/val/")
    print(f"  {OUTPUT_DIR}/test/")

if __name__ == "__main__":
    main()