import json
import cv2
with open('dataset/2020420_01_1.json','r') as f :
    img = cv2.imread('dataset/2020420_01_1.jpg')
    fully_height = img.shape[0]
    fully_width = img.shape[1]
    label_json = json.load(f)
    for instance in label_json['shapes']:
        x = (instance['points'][0][0] + instance['points'][2][0]) / (2 * fully_width)
        y = (instance['points'][0][1] + instance['points'][2][1]) / (2 * fully_height)
        width = abs((instance['points'][2][0] - instance['points'][0][0]) / fully_width)
        height = abs((instance['points'][2][1] - instance['points'][0][1]) / fully_height)
        print(x , y, width, height)
        print(fully_height, fully_width)
        print(instance['points'][0][0])
        print(instance['points'][1][0])
        print(instance['points'][2][0])
        print(instance['points'][3][0])
        print(instance['points'][0][1])
        print(instance['points'][1][1])
        print(instance['points'][2][1])
        print(instance['points'][3][1])

