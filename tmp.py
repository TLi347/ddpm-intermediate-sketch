# pip install Pillow
import PIL.Image as Image
# 以第一个像素为准，相同色改为透明
def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = (255,255,255,255)#要替换的颜色
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
    return img

if __name__ == '__main__':
    import glob
    import os
    fp = glob.glob('datasets/n02389026_horse/real/train/*.JPEG')
    for i in range(len(fp)):
        filename = fp[i]
        fileid = filename.split('/')[-1].split('.')[0]
        img=Image.open(filename)
        img=transparent_back(img)
        img.save(os.path.join('datasets/n02389026_horse/real/train', fileid+'.png'))