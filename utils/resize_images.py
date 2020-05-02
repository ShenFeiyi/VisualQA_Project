# 预处理输入图片
import os
import argparse
from PIL import Image

from VisualQA_Path import img


# 将image,resize成size型号
def resize_image(image, size):
    """Resize(compress, not crop) an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(input_dir, output_dir, size):
    # 输入路径为 train_img_path, val_img_path, test_img_path:
    # 包含三个文件夹, -训练集(82783), -验证集(40504), -测试集(81434), 每个文件夹下是图片
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir): # 浏览该目录下的子目录
        if not idir.is_dir(): # 若该目录不存在
            continue
        if not os.path.exists(output_dir+'/'+input_dir.split('/')[-1:][0]+'/'+idir.name): # e.g. '../datasets/img/train2014'
            os.makedirs(output_dir+'/'+input_dir.split('/')[-1:][0]+'/'+idir.name) # 建立输出目录
        
        images = os.listdir(idir.path) # 该文件夹下的文件（图片名）
        n_images = len(images)
        for iimage, image in enumerate(images):
            try:
                with open(os.path.join(idir.path, image), 'r+b') as f:
                    with Image.open(f) as img:
                        img = resize_image(img, size)
                        img.save(os.path.join(output_dir+'/'+input_dir.split('/')[-1:][0]+'/'+idir.name, image))
            except(IOError, SyntaxError) as e:
                print(f'ERROR `{e}` OCCURRED.')
            if (iimage+1) % 1000 == 0:
                print(f"[{iimage+1}/{n_images}] Resized the images and saved into '{output_dir+'/'+input_dir.split('/')[-1:][0]+'/'+idir.name}'.")

def main(params):

    input_dir = params['input_dir']
    output_dir = params['output_dir']
    image_size = [params['image_size'], params['image_size']]
    
    resize_images(input_dir, output_dir, image_size)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    #给定输入参数默认值
    parser.add_argument('--input_dir', type=str, default='/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/VisualQA/images',
                        help='directory for input images (unresized images)')

    parser.add_argument('--output_dir', type=str, default='../datasets',
                        help='directory for output images (resized images)')

    parser.add_argument('--image_size', type=int, default=224,
                        help='size of images after resizing')

    args = parser.parse_args()
    params = vars(args)
    '''
    params = {
        'input_dir':img, # image path
        'output_dir':'../datasets',
        'image_size':224 # vgg input size
        }
    main(params)

