import os
import time
import json
import ultility

def save_train_data(base_path):
    # Save file
    save_train_imgs_loc = os.path.join(base_path,"train_imgs.txt")
    save_classes_count_loc = os.path.join(base_path,"classes_count.txt")
    save_class_mapping_loc = os.path.join(base_path,"class_mapping.txt")
    if os.path.exists(save_train_imgs_loc):
        os.remove(save_train_imgs_loc)
    if os.path.exists(save_classes_count_loc):
        os.remove(save_classes_count_loc)
    if os.path.exists(save_class_mapping_loc):
        os.remove(save_class_mapping_loc)
    with open(save_train_imgs_loc, 'w') as f1, open(save_classes_count_loc, 'w') as f2, open(save_class_mapping_loc, 'w') as f3:
        json.dump(train_imgs, f1)
        json.dump(classes_count, f2)
        json.dump(class_mapping, f3)
    print("Done!")

def load_saved_data(base_path, imbalance=False):
    save_train_imgs_loc = os.path.join(base_path,"train_imgs.txt")
    save_classes_count_loc = os.path.join(base_path,"classes_count.txt")
    save_class_mapping_loc = os.path.join(base_path,"class_mapping.txt")
    with open(save_train_imgs_loc) as f1:
      train_imgs = json.load(f1)
    with open(save_classes_count_loc) as f2:
      classes_count = json.load(f2)
    with open(save_class_mapping_loc) as f3:
      class_mapping = json.load(f3)

    if imbalance==True:
        # Get random 1093 Sedan img from train list
        all_single_Sedan_list = [train_imgs[i] for i in range(len(train_imgs)) if (len(train_imgs[i]["bboxes"]) < 2) and (train_imgs[i]['bboxes'][0]['class'] == 'Sedan')]
        nums_img = len(all_single_Sedan_list)
        nums_select = 1093
        rm_list = np.random.choice(all_single_Sedan_list, nums_img - nums_select, replace=False).tolist()
        while len(rm_list) > 0:
          rm_item = rm_list.pop()
          if rm_item in train_imgs:
            train_imgs.remove(rm_item)
        # Update class count
        classes_count['Sedan'] = 1093
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    return train_imgs, classes_count, class_mapping
