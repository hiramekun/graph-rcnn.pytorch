import os
import shutil

data_dir = "./data/VOC2019_rel"
annotations_dir = data_dir + "/Annotations"
images_dir = data_dir + "/JPEGImages"

fnames_anno = os.listdir(annotations_dir)
fnames_img = os.listdir(images_dir)

use_anno = [os.path.splitext(img)[0] + ".xml" for img in fnames_img]

new_annotations_dir = data_dir + "/Annotations_new"
if not os.path.exists(new_annotations_dir):
    os.mkdir(new_annotations_dir)

use_img = []
for anno in use_anno:
    src = annotations_dir + "/" + anno
    if os.path.exists(src):
        shutil.copy(src, new_annotations_dir + "/" + anno)
        use_img.append(os.path.splitext(anno)[0] + ".jpg")

new_imgs_dir = data_dir + "/JPEGImages_new"
if not os.path.exists(new_imgs_dir):
    os.mkdir(new_imgs_dir)

for img in use_img:
    src = images_dir + "/" + img
    shutil.copy(src, new_imgs_dir + "/" + img)
