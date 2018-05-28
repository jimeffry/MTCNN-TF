import xml.etree.cElementTree as ET
tree = ET.parse("/home/lxy/Downloads/DataSet/VOC_Person/VOC2012/Annotations/2007_000664.xml")
root = tree.getroot()
for child_of_root in root:
    if child_of_root.tag == 'object':
        for child_item in child_of_root:
            print(child_item.tag) 
         
