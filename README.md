# 3d-bbox-detect
这个项目主要目的是通过微调Qwen2-VL-2B-Instruct模型以使其专门用于检测720*1280尺寸图片中的交通物品。
#### 使用方法
本训练预处理的是coco格式的数据集。先用./label/generation.py生成.jsonl文件，再用./sft/train_sft.py生成.json文件。最后运行sft.sh文件即可。