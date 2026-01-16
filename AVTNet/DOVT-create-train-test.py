import os
import shutil
import random
import glob


def split_dataset(rgb_dir, tactile_dir, test_rgb_dir, test_tactile_dir, split_ratio=0.25):
    """
    Args:
        rgb_dir: 原RGB文件夹路径 (训练集保留在此)
        tactile_dir: 原Tactile文件夹路径 (训练集保留在此)
        test_rgb_dir: 新建的测试集RGB路径
        test_tactile_dir: 新建的测试集Tactile路径
        split_ratio: 测试集占比 (例如 0.25 表示 3:1 划分)
    """

    # 1. 创建测试集目录
    if not os.path.exists(test_rgb_dir):
        os.makedirs(test_rgb_dir)
    if not os.path.exists(test_tactile_dir):
        os.makedirs(test_tactile_dir)

    # 2. 获取所有png文件
    # 假设文件名格式为:
    # RGB:     obj0_3_0_rgb_00001.png
    # Tactile: obj0_3_0_raw_00001.png

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    tactile_files = sorted(glob.glob(os.path.join(tactile_dir, "*.png")))

    print(f"扫描到 RGB 文件: {len(rgb_files)} 个")
    print(f"扫描到 Tactile 文件: {len(tactile_files)} 个")

    # 3. 建立配对关系
    # 提取共有ID作为key。根据你的例子，ID应该是去掉了中间的类型标识(_rgb_ 或 _raw_)
    # 逻辑：提取 objX_Y_Z 和 后面的数字后缀作为匹配键

    pairs = []

    # 构建一个辅助字典来查找
    # 键是: "obj0_3_0_00001", 值是: 完整路径
    tactile_map = {}
    for t_path in tactile_files:
        filename = os.path.basename(t_path)
        # 替换 _raw_ 为空字符，作为统一的key (根据具体命名规则调整)
        # 例子: obj0_3_0_raw_00001.png -> obj0_3_0_00001.png
        key = filename.replace("_raw_", "_")
        tactile_map[key] = t_path

    # 遍历RGB文件寻找匹配
    valid_pairs = []
    for r_path in rgb_files:
        filename = os.path.basename(r_path)
        # 例子: obj0_3_0_rgb_00001.png -> obj0_3_0_00001.png
        key = filename.replace("_rgb_", "_")

        if key in tactile_map:
            valid_pairs.append((r_path, tactile_map[key]))
        else:
            print(f"警告: 未找到匹配的 Tactile 文件: {filename}")

    total_pairs = len(valid_pairs)
    print(f"成功配对: {total_pairs} 组")

    if total_pairs == 0:
        print("错误: 没有找到匹配的文件，请检查命名规则或文件夹路径。")
        return

    # 4. 随机打乱并划分
    random.seed(42)  # 设置随机种子，保证结果可复现
    random.shuffle(valid_pairs)

    test_count = int(total_pairs * split_ratio)
    test_pairs = valid_pairs[:test_count]
    # 剩下的保留在原文件夹作为训练集，不需要操作

    print(f"即将移动 {len(test_pairs)} 组文件到测试集文件夹...")

    # 5. 移动文件到测试集目录
    for r_src, t_src in test_pairs:
        # 移动 RGB
        r_dst = os.path.join(test_rgb_dir, os.path.basename(r_src))
        shutil.move(r_src, r_dst)

        # 移动 Tactile
        t_dst = os.path.join(test_tactile_dir, os.path.basename(t_src))
        shutil.move(t_src, t_dst)

    print("--- 划分完成 ---")
    print(f"训练集 (原文件夹): {total_pairs - test_count} 组")
    print(f"测试集 (新文件夹): {test_count} 组")


# --- 配置区域 ---
if __name__ == "__main__":
    # 请在这里修改你的实际路径
    # 建议使用绝对路径，或者确保相对路径正确
    source_rgb = "/media/meiguiz/dataset2/AVTNet/DOVT-dataset/train/rgb"  # RGB 原文件夹
    source_tactile = "/media/meiguiz/dataset2/AVTNet/DOVT-dataset/train/tactile"  # Tactile 原文件夹

    dest_test_rgb = "/media/meiguiz/dataset2/AVTNet/DOVT-dataset/test/rgb"  # 测试集 RGB 保存位置
    dest_test_tactile = "/media/meiguiz/dataset2/AVTNet/DOVT-dataset/test/tactile"  # 测试集 Tactile 保存位置

    # 执行划分 (0.25 代表 25% 测试集，即 3:1)
    split_dataset(source_rgb, source_tactile, dest_test_rgb, dest_test_tactile, split_ratio=0.25)