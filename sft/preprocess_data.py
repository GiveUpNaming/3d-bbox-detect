import json
import os

def convert_to_qwen_format(input_file, output_file, image_base_path=""):
    """
    将自定义的jsonl数据转换为Qwen-VL SFT所需的格式。

    Args:
        input_file (str): 原始jsonl文件的路径。
        output_file (str): 输出jsonl文件的路径。
        image_base_path (str): 如果原始json中的image_file是相对路径，这里提供基础路径。
    """
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            try:
                original_item = json.loads(line)
                
                # 1. 解析 output_string
                # output_string 本身是一个字符串化的JSON，需要再次解析
                annotations = json.loads(original_item['output_string'])
                
                # 2. 构建助手的回答字符串
                assistant_parts = []
                for anno in annotations:
                    box = anno['box_2d']
                    label = anno['label']
                    
                    # 将 [0, 1] 范围的归一化坐标转换为 [0, 1000] 范围的整数坐标
                    x_min = box[0]
                    y_min = box[1]
                    x_max = box[2]
                    y_max = box[3]

                    # 确保坐标在 [0, 1000] 范围内
                    # x_min, y_min = max(0, x_min), max(0, y_min)
                    # x_max, y_max = min(1000, x_max), min(1000, y_max)
                    
                    # 构建Qwen-VL格式的字符串
                    # 您可以根据需要调整前缀文本，例如 "Here is a ..."
                    qwen_box_str = f"<ref>{label}</ref><box>({x_min},{y_min})({x_max},{y_max})></box>"
                    assistant_parts.append(qwen_box_str)
                
                # 如果有需要，可以加一个引导语
                assistant_response = "Here are the objects I found: " + " ".join(assistant_parts)

                # 3. 构建新的数据项
                new_item = {
                    # 将图片路径和基础路径结合
                    "image": os.path.join(image_base_path, original_item['image_file']),
                    "conversations": [
                        {
                            "from": "user",
                            "value": original_item['prompt']
                        },
                        {
                            "from": "assistant",
                            "value": assistant_response
                        }
                    ]
                }
                processed_data.append(new_item)

            except json.JSONDecodeError:
                print(f"警告：跳过无法解析的行: {line.strip()}")
            except Exception as e:
                print(f"处理行时出错: {line.strip()}\n错误: {e}")

    # 4. 写入新的json文件 (注意不是jsonl，是一个包含所有对象的列表的json文件)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(processed_data, f_out, indent=2, ensure_ascii=False)

    print(f"数据处理完成！已将 {len(processed_data)} 条数据写入 {output_file}")


# --- 使用示例 ---
# 假设您的jsonl文件名为 'my_data.jsonl'
# 并且图片位于 'dataset/' 目录下，而jsonl里的路径是 'images/train/...'
# 那么 image_base_path 应该是 'dataset'
# 如果jsonl里的路径已经是完整的，则 image_base_path 留空

# 运行预处理
convert_to_qwen_format(
    input_file='/home/users/di.feng/Projects/tufindtu/label_generation/val/processed_coco_traffic_val.jsonl',
    output_file='/home/users/di.feng/Projects/tufindtu/eval/processed_data_val.json',
    image_base_path=''  # 如果你的路径需要拼接，请修改这里
)