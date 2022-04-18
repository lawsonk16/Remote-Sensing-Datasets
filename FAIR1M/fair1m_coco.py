import os
import json
from tqdm import tqdm
import bs4
import lxml
from bs4 import BeautifulSoup as bs


def fair1m_cats():
    '''
    A list of object categories in the FAIR1M dataset
    Created by crawling through the data, making a list of category tags, 
    then re-ordering them by supercategory and hand-adding supercategories
    '''

    categories = [{'id': 1, 'name': 'Dry Cargo Ship', 'supercategory': 'Ship'},
                  {'id': 2, 'name': 'Engineering Ship', 'supercategory': 'Ship'},
                  {'id': 3, 'name': 'Motorboat', 'supercategory': 'Ship'},
                  {'id': 4, 'name': 'Liquid Cargo Ship', 'supercategory': 'Ship'},
                  {'id': 5, 'name': 'Warship', 'supercategory': 'Ship'},
                  {'id': 6, 'name': 'Passenger Ship', 'supercategory': 'Ship'},
                  {'id': 7, 'name': 'Tugboat', 'supercategory': 'Ship'},
                  {'id': 8, 'name': 'Fishing Boat', 'supercategory': 'Ship'},
                  {'id': 9, 'name': 'other-ship', 'supercategory': 'Ship'},

                  {'id': 10, 'name': 'Small Car', 'supercategory': 'Vehicle'},
                  {'id': 11, 'name': 'Van', 'supercategory': 'Vehicle'},
                  {'id': 12, 'name': 'Bus', 'supercategory': 'Vehicle'},
                  {'id': 13, 'name': 'Excavator', 'supercategory': 'Vehicle'},
                  {'id': 14, 'name': 'Tractor', 'supercategory': 'Vehicle'},
                  {'id': 15, 'name': 'Dump Truck', 'supercategory': 'Vehicle'},
                  {'id': 16, 'name': 'Cargo Truck', 'supercategory': 'Vehicle'},
                  {'id': 17, 'name': 'Truck Tractor', 'supercategory': 'Vehicle'},
                  {'id': 18, 'name': 'Trailer', 'supercategory': 'Vehicle'},
                  {'id': 19, 'name': 'other-vehicle', 'supercategory': 'Vehicle'},
                  
                  {'id': 20, 'name': 'Boeing737', 'supercategory': 'Airplane'},
                  {'id': 21, 'name': 'Boeing747', 'supercategory': 'Airplane'},
                  {'id': 22, 'name': 'Boeing777', 'supercategory': 'Airplane'},
                  {'id': 23, 'name': 'Boeing787', 'supercategory': 'Airplane'},
                  {'id': 24, 'name': 'ARJ21', 'supercategory': 'Airplane'},
                  {'id': 25, 'name': 'A220', 'supercategory': 'Airplane'},
                  {'id': 26, 'name': 'A321', 'supercategory': 'Airplane'},
                  {'id': 27, 'name': 'A330', 'supercategory': 'Airplane'},
                  {'id': 28, 'name': 'A350', 'supercategory': 'Airplane'},
                  {'id': 29, 'name': 'C919', 'supercategory': 'Airplane'},
                  {'id': 30, 'name': 'other-airplane', 'supercategory': 'Airplane'},

                  {'id': 31, 'name': 'Baseball Field', 'supercategory': 'Court'},
                  {'id': 32, 'name': 'Football Field', 'supercategory': 'Court'},
                  {'id': 33, 'name': 'Tennis Court', 'supercategory': 'Court'},
                  {'id': 34, 'name': 'Basketball Court', 'supercategory': 'Court'},

                  {'id': 35, 'name': 'Intersection', 'supercategory': 'Road'},
                  {'id': 36, 'name': 'Roundabout', 'supercategory': 'Road'},
                  {'id': 37, 'name': 'Bridge', 'supercategory': 'Road'}]
    return categories

def fair1m_coco_ims_cats_anns(xml_fp):

    # intialize key variables
    images = []
    ann_count = 0
    annotations = []
    categories = fair1m_cats()

    for a in tqdm(os.listdir(xml_fp)):
        # get path to label file
        label_fp = xml_fp + a
        # extract data from file
        # Get content with beautiful soup
        content = []
        # Read the XML file
        with open(label_fp, "r") as file:
            # Read each line in the file, readlines() returns a list of lines
            content = file.readlines()
            # Combine the lines in the list into a string
            content = "".join(content)
            bs_content = bs(content, "lxml")

        ## get coco stuff ##

        # pull out image info
        im_name = bs_content.find('filename').text
        im_id = int(im_name.split('.')[0])

        im_w = int(bs_content.find('size').find('width').text)
        im_h = int(bs_content.find('size').find('height').text)

        # construct image annotations
        im_info = {
            "id": im_id, 
            "width": im_w, 
            "height": im_h, 
            "file_name": im_name, 
            "license": 1
        }
        images.append(im_info)

        # process objects on image
        objects = bs_content.find('objects')

        for o in objects:
            if len(o) > 1:
                # get object name
                ob_n = o.find('name').text

                coord_type = o.find('coordinate').text

                if coord_type != 'pixel':
                    print(coord_type)

                # get object category id or create it
                cat_exists = False
                for c in categories:
                    if c['name'] == ob_n:
                        ann_cat_id = c['id']
                        cat_exists = True
                if not cat_exists:
                    print('Category down!', ob_n)
              

                # get points
                points = o.find('points')
                pt_list = []
                for p in points:
                    try:
                        pt_list.append(p.text)
                    except:
                        continue
                pts = []
                for p in pt_list:
                    (x,y) = p.split(',')
                    x = int(float(x))
                    y = int(float(y))
                    pts.append((x,y))
                
                # get coco style bbox
                xs = [b[0] for b in pts]
                ys = [b[1] for b in pts]
                xs, ys
                x1 = min(xs)
                y1 = min(ys)
                w = max(xs) - x1
                h = max(ys) - y1
                
                ann = {
                      "id": ann_count, 
                      "image_id": im_id, 
                      "category_id": ann_cat_id, 
                      "area": None, 
                      "segmentation": pts,
                      "bbox": [x1, y1, w, h],
                      "iscrowd": 0
                      }
                annotations.append(ann)
                ann_count += 1

    return images, categories, annotations

def fair1m_json(json_path, xml_fp):

    # ensure that no duplicate content is created
    if os.path.exists(json_path):
        os.remove(json_path)
    
    # get images, categories, and annotations
    images, categories, annotations = fair1m_coco_ims_cats_anns(xml_fp)

    # load the coco json
    coco_content = {
        'images' : images,
        'categories': categories,
        'annotations': annotations,
        'license': {"id": 1, "name": 'Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.', "url": 'https://creativecommons.org/licenses/by-nc-sa/3.0/'},
        'info': {"year": 2021, "version": '1.0', "description": 'FAIR1M Challenge Dataset 2021', "paper": 'https://arxiv.org/abs/2103.05569v2', "url": 'http://gaofen-challenge.com/indexpage', "date_created": '2021'}
        }
    
    # save the content
    with open(json_path, 'w') as f:
        json.dump(coco_content, f)

    return json_path