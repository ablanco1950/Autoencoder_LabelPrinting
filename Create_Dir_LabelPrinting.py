import os
import shutil
output_dir="label_data"

if  os.path.exists(output_dir):shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + "\\train")
os.mkdir(output_dir+ "\\test")
os.mkdir(output_dir + "\\train\\bad_label")
os.mkdir(output_dir + "\\train\\good_label")
os.mkdir(output_dir + "\\test\\bad_label")
os.mkdir(output_dir + "\\test\\good_label")
