import nibabel as nib
import numpy as np
from pathlib2 import Path
from scipy import ndimage
import cv2
import multiprocessing as mp
import click

@click.command()
@click.option('-d', '--data', help='kits19 data path',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True), required=True)
@click.option('-o', '--output', help='output npy file path',
              type=click.Path(dir_okay=True, resolve_path=True), required=True)
def conversion_all(data, output):
    data = Path(data)
    output = Path(output)

    cases = sorted([d for d in data.iterdir() if d.is_dir()])
    pool = mp.Pool()
    pool.map(conversion, zip(cases, [output] * len(cases)))
    pool.close()
    pool.join()

def conversion(data):
    case, output = data
    vol_nii = nib.load(str(case / 'imaging.nii.gz'))
    vol = vol_nii.get_fdata()
    vol = resample(vol,vol_nii.header['pixdim'][1:4])
    vol = clip_hu(vol)

    imaging_dir = output / case.name / 'imaging'
    if not imaging_dir.exists():
        imaging_dir.mkdir(parents=True)
    if len(list(imaging_dir.glob('*.npy'))) != vol.shape[0] *2:
        for i in range(vol.shape[0]):
            np.save(str(imaging_dir / f'{i:03}.npy'), vol[i])
            np.save(str(imaging_dir / f'{i:03}_sobel.npy'), get_sobel(vol[i]))

    segmentation_file = case / 'segmentation.nii.gz'
    if segmentation_file.exists():
        seg_nii = nib.load(str(case / 'segmentation.nii.gz'))
        seg = seg_nii.get_fdata()
        seg = resample(seg,seg_nii.header['pixdim'][1:4])


        segmentation_dir = output / case.name / 'segmentation'
        if not segmentation_dir.exists():
            segmentation_dir.mkdir(parents=True)
        if len(list(segmentation_dir.glob('*.npy'))) != seg.shape[0]:
            for i in range(seg.shape[0]):
                np.save(str(segmentation_dir / f'{i:03}.npy'), seg[i])
                # np.save(str(segmentation_dir / f'{i:03}_tag.npy'), tag[i])

    affine_dir = output / case.name
    if not affine_dir.exists():
        affine_dir.mkdir(parents=True)
    affine = vol_nii.affine
    np.save(str(affine_dir / 'affine.npy'), affine)
    
def resample(vol,pixdim):
    # resample to 3.22*1.62*1.62 mm with 128*248*248 [z,x,y]
    bound_z = 3.22*128
    bound_x = 1.62*248
    bound_y = 1.62*248
    z_true = np.abs(pixdim[0]*vol.shape[0])
    x_true = np.abs(pixdim[1]*vol.shape[1])
    y_true = np.abs(pixdim[2]*vol.shape[2])

#     if len(vol.shape)<3:
#         print(vol_nii.header)
#     print(vol.shape)
    if z_true>bound_z:
        z_crop = np.abs(int((z_true - bound_z)/2/pixdim[0]))
#         print(z_crop)
        vol = vol[z_crop:-(z_crop+1),:,:]
#         print(vol_nii_data.shape[0])
#         print(2,vol.shape)
        z_factor = 128/vol.shape[0]
    else:
        z_factor = z_true/3.22/vol.shape[0]
        
    if x_true>bound_x:
        x_crop = np.abs(int((x_true - bound_x)/2/pixdim[1]))
        vol = vol[:,x_crop:-(x_crop+1),:]
        x_factor = 248/vol.shape[1]
    else:
#         print(x_true)
        x_factor = x_true/1.62/vol.shape[1]
        
    if y_true>bound_y:
        y_crop = np.abs(int((y_true - bound_y)/2/pixdim[2]))
        vol = vol[:,:,y_crop:-(y_crop+1)]
        y_factor = 248/vol.shape[2]
    else:
        y_factor = y_true/1.62/vol.shape[2]


    img_3D = ndimage.zoom(vol, (z_factor, x_factor, y_factor))
    return img_3D

def clip_hu(img_3D):
    hu_max = 304
    hu_min = -79
    img_3D = np.clip(img_3D, hu_min, hu_max)
    return img_3D

def get_sobel(img):
    sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1)
    abx = cv2.convertScaleAbs(sobel_x)
    aby = cv2.convertScaleAbs(sobel_y)
    result_sobel = cv2.addWeighted(abx,0.5,aby,0.5,0)
    return result_sobel


    



if __name__ == '__main__':
    conversion_all()
