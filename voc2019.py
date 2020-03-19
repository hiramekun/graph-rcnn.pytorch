import collections
import json
import os
import sys

from structures.bounding_box import BoxList
from torchvision.datasets.vision import VisionDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image


class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)

        base_dir = 'VOC2019_rel'
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages_new')
        annotation_dir = os.path.join(voc_root, 'Annotations_new')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        file_names = [os.path.splitext(os.path.basename(x))[0] for x in os.listdir(image_dir)]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        self.info = json.load(
            open(os.path.join("./datasets/vg_bm/VG-SGG-dicts.json"), 'r'))
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k: self.class_to_ind[k])
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target, labels = self._get_target(index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target.add_field("labels", labels)
        target = target.clip_to_image(remove_empty=False)

        return img, target, self.images[index]

    def __len__(self):
        return len(self.images)

    def _get_target(self, index):
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        objs = target['annotation']['object']
        size = target['annotation']['size']

        bboxs = []
        classes = []
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            label = obj['name']
            bbox = obj['bndbox']
            xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(
                bbox['ymax'])
            bboxs.append((xmin, ymin, xmax, ymax))
            classes.append(label)

        target_raw = BoxList(bboxs, (int(size['width']), int(size['height'])), mode='xyxy')
        return target_raw, classes

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
