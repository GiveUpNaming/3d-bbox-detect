import json

# 读取 JSON 文件
with open('/home/users/di.feng/Projects/tufindtu/tmp.json', 'r') as file:
    data = json.load(file)

box2d_vectors = []

for frame in data['frames']:
    for obj in frame['objects']:
        if 'box2d' in obj:
            box2d = obj['box2d']
            vector = [box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']]
            box2d_vectors.append(vector)
with open('/home/users/di.feng/Projects/tufindtu/tmp1.txt', 'w') as output_file:
    for vector in box2d_vectors:
        output_file.write(f"{vector[0]} {vector[1]} {vector[2]} {vector[3]}\n")
