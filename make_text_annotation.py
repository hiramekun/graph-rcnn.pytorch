import os
import shutil

data_dir = "./data/VOC2019_rel"
annotations_dir = data_dir + "/Annotations_new"
text_dir = data_dir + "/Texts"

if os.path.exists(text_dir):
    shutil.rmtree(text_dir)

os.mkdir(text_dir)
for anno in os.listdir(annotations_dir):
    basename = os.path.splitext(anno)[0]
    with open(text_dir + "/" + basename + ".txt", "w") as f:
        if basename.startswith("2019"):
            f.write("man wearing helmet\nman wearing glove\nman wearing glove")
        else:
            str_arr = basename.split("+")
            write_str = ""

            if str_arr[0].startswith("mask"):
                # if str_arr[1].startswith("glass"):
                #     write_str += "man wearing glass\n"
                if len(str_arr) > 2 and str_arr[2].startswith("long"):
                    write_str += "man wearing helmet\n"

            if str_arr[0].startswith("visorup"):
                write_str += "man wearing helmet"
                pass
                # if str_arr[1].startswith("glass"):
                    # write_str += "man wearing glass\n"
            elif str_arr[0].startswith("visor"):
                write_str += "man wearing helmet"
                pass
                # if str_arr[1].startswith("glass"):
                #     write_str += "man wearing glass\n"

            if len(write_str) == 0:
                write_str = "other"
            f.write(write_str)
