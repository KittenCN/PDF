from PIL import Image
import os

templateadd = r'./template/'
pre_processadd = r'./pre_process/'
post_processadd = r'./post_process/'

if __name__ == "__main__":
    tempfiles = os.listdir(templateadd)
    prefiles = os.listdir(pre_processadd)
    jpgfiles = []
    for tmp in tempfiles:
        if tmp.endswith('.jpg'):
            jpgfiles.append(templateadd + tmp)
    jpgfiles.sort()
    for index, pre in enumerate(prefiles):
        source = []
        if pre.endswith('.jpg'):
            _jpgfiles = jpgfiles.copy()
            _jpgfiles.append(pre_processadd + pre)
            output = Image.open(_jpgfiles[0])
            _jpgfiles.pop(0)
            for jpg in _jpgfiles:
                _jpgfile = Image.open(jpg)
                if _jpgfile.mode == "RGB":
                    _jpgfile = _jpgfile.convert("RGB")
                source.append(_jpgfile)
            output.save(post_processadd + str(index) + ".pdf", "pdf", save_all=True, append_images=source)


                    