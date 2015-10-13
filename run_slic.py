import PIL.Image
import glob
import random
import subprocess
import os.path
from .utils import *


def run_slic(img_loc, num=500, weight_factor=50):
    w, h = PIL.Image.open(img_loc).size
    subprocess.call(['./test_slic', img_loc, str(num), str(weight_factor)])
    base = img_loc.rsplit('.', 1)[0]
    regions = np.frombuffer(open(base + '_regions.dat', 'rb').read(), 'int32').reshape((h, w))
    contours = np.frombuffer(open(base + '_contours.dat', 'rb').read(), 'uint8').reshape((h, w)) - 48
    return (regions, contours)


data = dict()
for img_loc in glob.glob('BSDS500/*.jpg'):
    base = os.path.basename(img_loc).replace('.jpg', '')
    
    slic_data = run_slic(img_loc, 1000, 40)
    bsds_data = random.choice(get_bsds_data('BSDS500/%s.mat' % base))
    data[base] = {
        'underseg': undersegmentation_error(slic_data[0], bsds_data['regions']), 
        'br': boundary_recall_error(slic_data[1], bsds_data['contours'])
    }
    print(base, 'done')

print(np.mean([d['br'] for d in data.values()]), np.mean([d['underseg'] for d in data.values()]))