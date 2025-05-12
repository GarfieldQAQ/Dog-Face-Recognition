import os
import xml.etree.ElementTree as ET
import glob

def convert_coordinates(size, box):
    """将XML中的边界框坐标转换为YOLO格式"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_xml_to_yolo(xml_path, class_mapping):
    """转换单个XML文件到YOLO格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    txt_path = xml_path.replace('Annotations', 'Labels').replace('.xml', '.txt')
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as txt_file:
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue
            class_id = class_mapping[class_name]
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            bb = convert_coordinates((width, height), (xmin, ymin, xmax, ymax))
            txt_file.write(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

def main():
    class_mapping = {
        'face': 0,
        'mouse_bite': 1,
        'open_circuit': 2,
        'short': 3,
        'spur': 4,
        'spurious_copper': 5
    }
    xml_files = glob.glob('./*.xml')
    for xml_file in xml_files:
        try:
            convert_xml_to_yolo(xml_file, class_mapping)
            print(f"成功转换: {xml_file}")
        except Exception as e:
            print(f"转换失败 {xml_file}: {str(e)}")

if __name__ == "__main__":
    main()