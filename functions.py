import SimpleITK as sitk
import numpy as np
import math
import csv
# from visualization import voxel2points
import os
from multiprocessing import Pool
import time

def progress(percent, width=50):
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")
    print('\r%s %.2f%%' % (show_str, percent))

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
                int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
                int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def start_points(my_shape):
    z_L = my_shape[0]  # int
    y_L = my_shape[1]
    x_L = my_shape[2]
    block_size = 96.0

    zyx_L = [z_L, y_L, x_L]
    start = []
    for L in zyx_L:
        sub_start = []
        num_block = int(math.ceil(L/block_size))
        if num_block > 1:
            stride = round((L-block_size)/(num_block-1))
            for i_s in range(int(num_block)-1):
                tiny_start = int(stride*i_s)
                sub_start.append(tiny_start)
        sub_start.append(int(L-block_size))
        start.append(sub_start)

    return start

def load_annotations(file_path):
    csv_file = csv.reader(open(file_path, 'r'))
    annotations = {}
    for i, ele in enumerate(csv_file):
        if i>0:
            if ele[0] not in annotations.keys():
                annotations[ele[0]] = []
            va = []
            for ele_va in ele[1:]:
                va.append(float(ele_va))
            annotations[ele[0]].append(va)
    return annotations

def worldToVoxelCoord_cor(worldCoord, origin, spacing):
    worldCoord = np.array(worldCoord, dtype=np.float)
    origin = np.array(origin, dtype=np.float)
    spacing = np.array(spacing, dtype=np.float)
    stretchedVoxelCoord = worldCoord - origin
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def worldToVoxelCoord_l(worldCoord, spacing):
    worldCoord = np.array(worldCoord, dtype=np.float)
    spacing = np.array(spacing, dtype=np.float)
    voxelCoord = worldCoord / spacing
    return voxelCoord

def find_block(ann_voxel, starts):
    # ann_voxel: [cz, cy, cx, dz, dy, dx, label]
    coor = ann_voxel[:3]
    block_start = []
    for i_d in range(3):
        starts_d = starts[i_d]
        num_block = len(starts_d)
        if coor[i_d] >= starts_d[-1]:
            block_start.append(starts_d[-1])
        else:
            for i_b in range(num_block-1):
                if coor[i_d]>=starts_d[i_b] and coor[i_d]<starts_d[i_b+1]:
                    block_start.append(starts_d[i_b])
    ann_voxel_block = ann_voxel
    ann_voxel_block[:3] -= block_start
    return block_start, ann_voxel_block


def voxel2points(voxel, spacing, file_path, threshold, spacing_expand, origin = np.zeros(3)):
    f = open(file_path, 'w')

    Z = voxel.shape[0]
    Y = voxel.shape[1]
    X = voxel.shape[2]
    z_spacing = spacing[0]
    y_spacing = spacing[1]
    x_spacing = spacing[2]

    z_origin = origin[0]
    y_origin = origin[1]
    x_origin = origin[2]


    for i in range(Z):
        percent = (i+1.0)*100.0/Z
        progress(percent)
        for j in range(Y):
            for k in range(X):
                x = k*x_spacing + x_origin
                y = j*y_spacing + y_origin
                z = i*z_spacing*spacing_expand + z_origin
                value = voxel[i,j,k]
                if value > threshold:
                    f.write(str(x)+' '+str(y)+' '+str(z)+' '+str(value)+'\n')

    f.close()


def add_points(points, file_path, spacing_expand):
    f = open(file_path, 'a')
    num_points = points.shape[0]
    for i in range(num_points):
        point = points[i,:]
        f.write(str(point[2])+' '+str(point[1])+' '+str(point[0]*spacing_expand)+' '+str(point[3])+'\n')
        f.write(str(point[2]+2) + ' ' + str(point[1]) + ' ' + str(point[0] * spacing_expand) + ' ' + str(point[3]) + '\n')
        f.write(str(point[2]) + ' ' + str(point[1]+2) + ' ' + str(point[0] * spacing_expand) + ' ' + str(point[3]) + '\n')
        f.write(str(point[2]) + ' ' + str(point[1]) + ' ' + str(point[0] * spacing_expand+2) + ' ' + str(point[3]) + '\n')
        f.write(str(point[2]-2) + ' ' + str(point[1]) + ' ' + str(point[0] * spacing_expand) + ' ' + str(point[3]) + '\n')
        f.write(str(point[2]) + ' ' + str(point[1]-2) + ' ' + str(point[0] * spacing_expand) + ' ' + str(point[3]) + '\n')
        f.write(str(point[2]) + ' ' + str(point[1]) + ' ' + str(point[0] * spacing_expand-2) + ' ' + str(point[3]) + '\n')
    f.close()

def save_label(anns, idx, file_index, out_path):
    out_label_path = out_path + '/' + str(file_index) + str(idx) + '_label.npy'
    if not os.path.exists(out_label_path):
        output = -1*np.ones((24, 24, 24, 3, 7), dtype=np.float)
        output[:,:,:,:,0] = 0   # background
        label_dict = {'1':1, '5':2, '31':3, '32':4}
        for ann in anns:
            cz = ann[0]
            cy = ann[1]
            cx = ann[2]
            dz = ann[3]
            dy = ann[4]
            dx = ann[5]
            label = ann[6]

            c_zyx = [cz, cy, cx]
            one_center = []
            for d in range(3):
                cor = c_zyx[d]
                i_idx = int(round(np.abs(cor-1.5)/4.0))
                one_center.append(i_idx)
            label = label_dict[str(int(round(label)))]
            output[one_center[0], one_center[1], one_center[2], :, :] = [label, cz, cy, cx, dz, dy, dx]
        np.save(out_label_path, output)


def save_one_block(idx, imgs, label_start, file_index, out_path):
    out_block_path = out_path + '/' + str(file_index) + str(idx) + '_data.npy'
    if not os.path.exists(out_block_path):
        block_size = 96
        # label_start: {'starts':, 'anns':}
        starts = label_start['starts']
        z_start = starts[0]
        y_start = starts[1]
        x_start = starts[2]
        block = imgs[z_start:z_start+block_size, y_start:y_start+block_size, x_start:x_start+block_size]

        np.save(out_block_path, block)

        anns = label_start['anns']
        # anns: [cz, cy, cx, dz, dy, dx, label]
        save_label(anns, idx, file_index, out_path)

def windows(imgs, my_type = 'lung'):
    assert my_type in ['lung', 'column']

    if my_type == 'lung':
        lungwin = np.array([1800.0, -500.0])  # windows width and position
    else:
        lungwin = np.array([300.0, 40.0])

    min_value = lungwin[1] - (lungwin[0]/2.0)
    newimgs = (imgs - min_value) / lungwin[0]
    newimgs[newimgs<0] = 0
    newimgs[newimgs>1] = 1
    newimgs = newimgs * 255.0
    return newimgs

def my_pad(imgs):
    pad_value = 170.0
    block_size = 96
    l_z = imgs.shape[0]
    pad = [[0, block_size-l_z], [0,0], [0,0]]
    imgs = np.pad(imgs, pad, 'constant', constant_values=pad_value)
    return imgs


def start_ann_mapping(starts, annotations, itk_resampled):
    origin = itk_resampled.GetOrigin()  # xyz
    spacing = itk_resampled.GetSpacing()  # xyz
    origin = origin[::-1]  # zyx
    spacing = spacing[::-1]  # zyx

    block_start_ann_list = []
    # ann_world: cx, cy, cz, dx, dy, dz, label
    for ann_world in annotations:
        c_xyz_world = ann_world[:3]  # xyz
        c_zyx_world = c_xyz_world[::-1]  # zyx
        c_zyx_voxel = worldToVoxelCoord_cor(c_zyx_world, origin, spacing)  # zyx
        l_xyz_world = ann_world[3:6]  # xyz
        l_zyx_world = l_xyz_world[::-1]  # zyx
        l_zyx_voxel = worldToVoxelCoord_l(l_zyx_world, spacing)  # zyx

        new_ann = np.concatenate((c_zyx_voxel, l_zyx_voxel), axis=0)
        new_ann = np.concatenate((new_ann, [ann_world[-1]]), axis=0)  # [cz,cy,cx,dz,dy,dx,label] voxel

        block_start, ann_voxel_block = find_block(new_ann, starts)
        block_start_ann_list.append([block_start, ann_voxel_block])  # [[block_start, ann_voxel_block],...]

    start_ann_dict = {}
    for start_ann in block_start_ann_list:
        b_start, b_ann = start_ann
        str_s = str(int(round(b_start[0]))) + '_' + str(int(round(b_start[1]))) + '_' + str(int(round(b_start[2])))
        if str_s not in start_ann_dict.keys():
            start_ann_dict[str_s] = []
        start_ann_dict[str_s].append(b_ann)

    label_start_list = []  # [{'starts':, 'anns':}, ...]
    for key in start_ann_dict.keys():
        start_digits = key.split('_')
        inner_dict = {'starts': [int(start_digits[0]), int(start_digits[1]), int(start_digits[2])],
                      'anns': start_ann_dict[key]}
        label_start_list.append(inner_dict)
    return label_start_list


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))  # xyz -> zyx
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

def sample_data(data_label, npoints):
    num_points = data_label.shape[0]
    if num_points >= npoints:
        idx = np.array(range(num_points))
        np.random.shuffle(idx)
        block_data_label = data_label[idx[0:npoints]]
        return block_data_label
    else:
        print('upsampling')
        num_dul = npoints-num_points
        idx = np.random.choice(num_points, num_dul)
        data_label_dul = data_label[idx]
        block_data_label = np.concatenate((data_label, data_label_dul), axis=0)
        return block_data_label

def calculate_starts(total_len, per_len):
    starts_d = []
    num_starts = int(total_len / per_len)
    for i_start in range(num_starts):
        starts_d.append(i_start * per_len)
    if (total_len - num_starts * per_len) > 0.2:
        starts_d.append(total_len - per_len)
    return starts_d


def create_block_data_label5(data_label, length, npoints, out_folder_path, ct_index, flag_save=0, flag_show=1):
    # data_label, shape=(n,8), z,y,x,ct,l1,l2,l3,l4

    # normalize coordinates to original points
    min_zyx = np.min(data_label[:,0:3], axis=0)
    data_label[:,0:3] -= min_zyx

    starts_z = calculate_starts(np.max(data_label[:,0]), length)
    starts_y = calculate_starts(np.max(data_label[:, 1]), length)
    starts_x = calculate_starts(np.max(data_label[:, 2]), length)

    zz, yy, xx = np.meshgrid(starts_z, starts_y, starts_x)
    zz = np.expand_dims(zz.transpose(1, 0, 2), axis=-1)  # shape=(zs,ys,xs)
    yy = np.expand_dims(yy.transpose(1, 0, 2), axis=-1)
    xx = np.expand_dims(xx.transpose(1, 0, 2), axis=-1)

    ss = np.concatenate((zz, yy), axis=-1)
    ss = np.concatenate((ss, xx), axis=-1)  # shape = (zs,ys,xs,3)

    ss = np.reshape(ss, (ss.shape[0]*ss.shape[1]*ss.shape[2], ss.shape[3]))  # shape=(n,3)



    block_mask = (data_label[:,0]>=ss[0, 0]) & (data_label[:,0]<(ss[0, 0]+length)) & \
                 (data_label[:,1]>=ss[0, 1]) & (data_label[:,1]<(ss[0, 1]+length)) & \
                 (data_label[:,2]>=ss[0, 2]) & (data_label[:,2]<(ss[0, 2]+length))
    block_data_label = data_label[block_mask]
    block_data_label = sample_data(block_data_label, npoints)
    block_data_label = np.expand_dims(block_data_label, axis=0) # shape=(1,npoints,8)


    i_ss = 0
    all_ss = len(ss)
    for start in ss[1:]:
        if flag_show:
            i_ss += 1
            progress(i_ss*100.0/all_ss)
        t_block_mask = (data_label[:,0]>=start[0]) & (data_label[:,0]<(start[0]+length)) & \
                 (data_label[:,1]>=start[1]) & (data_label[:,1]<(start[1]+length)) & \
                 (data_label[:,2]>=start[2]) & (data_label[:,2]<(start[2]+length))
        t_block_data_label = data_label[t_block_mask]
        t_block_data_label = sample_data(t_block_data_label, npoints)
        t_block_data_label = np.expand_dims(t_block_data_label, axis=0)
        block_data_label = np.concatenate((block_data_label, t_block_data_label), axis=0)  # shape=(B,npoints,8)

    new_block_data_label = np.zeros((block_data_label.shape[0], npoints, 4+4))  # data+label

    max_zyx = np.max(data_label[:, 0:3], axis=0)  # shape = (3,)
    new_block_data_label[:,:,0:3] = block_data_label[:,:,0:3]/max_zyx  # global

    new_block_data_label[:,:,3] = block_data_label[:,:,3]/255.0  # ct value

    new_block_data_label[:,:,4:] = block_data_label[:,:,4:]  # labels




    if flag_save:
        save_block_data_label = np.zeros(block_data_label.shape)
        save_block_data_label[:,:,0] = new_block_data_label[:,:,2]
        save_block_data_label[:,:,1] = new_block_data_label[:,:,1]
        save_block_data_label[:,:,2] = new_block_data_label[:,:,0]
        save_block_data_label[:,:,3] = new_block_data_label[:,:,3]
        save_block_data_label[:,:,4:] = new_block_data_label[:,:,4:]
        sub_block_folder_path = out_folder_path + '/' + ct_index
        if not os.path.exists(sub_block_folder_path):
            os.makedirs(sub_block_folder_path)

        for i_block in range(save_block_data_label.shape[0]):
            file_path = sub_block_folder_path + '/' + str(i_block) + '.txt'
            np.savetxt(file_path, save_block_data_label[i_block,:,:])

    return new_block_data_label


def create_block_data_label4(data_label, length, npoints, out_folder_path, ct_index, flag_save=0):
    # data_label, shape=(n,8), z,y,x,ct,l1,l2,l3,l4

    # normalize coordinates to original points
    min_zyx = np.min(data_label[:,0:3], axis=0)
    data_label[:,0:3] -= min_zyx

    starts_z = calculate_starts(np.max(data_label[:,0]), length)
    starts_y = calculate_starts(np.max(data_label[:, 1]), length)
    starts_x = calculate_starts(np.max(data_label[:, 2]), length)

    zz, yy, xx = np.meshgrid(starts_z, starts_y, starts_x)
    zz = np.expand_dims(zz.transpose(1, 0, 2), axis=-1)  # shape=(zs,ys,xs)
    yy = np.expand_dims(yy.transpose(1, 0, 2), axis=-1)
    xx = np.expand_dims(xx.transpose(1, 0, 2), axis=-1)

    ss = np.concatenate((zz, yy), axis=-1)
    ss = np.concatenate((ss, xx), axis=-1)  # shape = (zs,ys,xs,3)

    ss = np.reshape(ss, (ss.shape[0]*ss.shape[1]*ss.shape[2], ss.shape[3]))  # shape=(n,3)

    block_mask = (data_label[:,0]>=ss[0, 0]) & (data_label[:,0]<(ss[0, 0]+length)) & \
                 (data_label[:,1]>=ss[0, 1]) & (data_label[:,1]<(ss[0, 1]+length)) & \
                 (data_label[:,2]>=ss[0, 2]) & (data_label[:,2]<(ss[0, 2]+length))
    block_data_label = data_label[block_mask]
    block_data_label = sample_data(block_data_label, npoints)
    block_data_label = np.expand_dims(block_data_label, axis=0) # shape=(1,npoints,8)

    i_process = 0
    total_process = len(ss[1:])
    for start in ss[1:]:
        i_process += 1
        progress(i_process*100.0/total_process)
        t_block_mask = (data_label[:,0]>=start[0]) & (data_label[:,0]<(start[0]+length)) & \
                 (data_label[:,1]>=start[1]) & (data_label[:,1]<(start[1]+length)) & \
                 (data_label[:,2]>=start[2]) & (data_label[:,2]<(start[2]+length))
        t_block_data_label = data_label[t_block_mask]
        t_block_data_label = sample_data(t_block_data_label, npoints)  # shape=(npoint, 8)
        if np.sum(t_block_data_label[:,-4:])>0.0:
            t_block_data_label = np.expand_dims(t_block_data_label, axis=0)
            block_data_label = np.concatenate((block_data_label, t_block_data_label), axis=0)  # shape=(B,npoints,8)

    new_block_data_label = np.zeros((block_data_label.shape[0], npoints, 7+4))  # data+label

    min_block_zyx = np.min(block_data_label[:,:,0:3], axis=1)  # shape = (B,3)
    min_block_zyx = np.expand_dims(min_block_zyx, axis=1)  # shape = (B,1,3)
    new_block_data_label[:,:,0:3] = block_data_label[:,:,0:3]-(min_block_zyx+length/2.0)  # local geometry

    new_block_data_label[:,:,3] = block_data_label[:,:,3]/255.0  # ct value

    max_zyx = np.max(data_label[:,0:3], axis=0)  # shape = (3,)
    new_block_data_label[:,:,4:7] = block_data_label[:,:,0:3]/max_zyx   # global geometry
    new_block_data_label[:,:,7:] = block_data_label[:,:,4:]  # labels


    if flag_save:
        save_block_data_label = np.zeros(block_data_label.shape)
        save_block_data_label[:,:,0] = new_block_data_label[:,:,6]
        save_block_data_label[:,:,1] = new_block_data_label[:,:,5]
        save_block_data_label[:,:,2] = new_block_data_label[:,:,4]
        save_block_data_label[:,:,3] = new_block_data_label[:,:,3]
        save_block_data_label[:,:,4:] = new_block_data_label[:,:,7:]
        sub_block_folder_path = out_folder_path + '/' + ct_index
        if not os.path.exists(sub_block_folder_path):
            os.makedirs(sub_block_folder_path)

        for i_block in range(save_block_data_label.shape[0]):
            file_path = sub_block_folder_path + '/' + str(i_block) + '.txt'
            np.savetxt(file_path, save_block_data_label[i_block,:,:])

    return new_block_data_label



def create_block_data_label3(data_label, length, npoints, out_folder_path, ct_index, flag_save=0):
    # data_label, shape=(n,8), z,y,x,ct,l1,l2,l3,l4

    # normalize coordinates to original points
    min_zyx = np.min(data_label[:,0:3], axis=0)
    data_label[:,0:3] -= min_zyx

    starts_z = calculate_starts(np.max(data_label[:,0]), length)
    starts_y = calculate_starts(np.max(data_label[:, 1]), length)
    starts_x = calculate_starts(np.max(data_label[:, 2]), length)

    zz, yy, xx = np.meshgrid(starts_z, starts_y, starts_x)
    zz = np.expand_dims(zz.transpose(1, 0, 2), axis=-1)  # shape=(zs,ys,xs)
    yy = np.expand_dims(yy.transpose(1, 0, 2), axis=-1)
    xx = np.expand_dims(xx.transpose(1, 0, 2), axis=-1)

    ss = np.concatenate((zz, yy), axis=-1)
    ss = np.concatenate((ss, xx), axis=-1)  # shape = (zs,ys,xs,3)

    ss = np.reshape(ss, (ss.shape[0]*ss.shape[1]*ss.shape[2], ss.shape[3]))  # shape=(n,3)



    block_mask = (data_label[:,0]>=ss[0, 0]) & (data_label[:,0]<(ss[0, 0]+length)) & \
                 (data_label[:,1]>=ss[0, 1]) & (data_label[:,1]<(ss[0, 1]+length)) & \
                 (data_label[:,2]>=ss[0, 2]) & (data_label[:,2]<(ss[0, 2]+length))
    block_data_label = data_label[block_mask]
    block_data_label = sample_data(block_data_label, npoints)
    block_data_label = np.expand_dims(block_data_label, axis=0) # shape=(1,npoints,8)

    for start in ss[1:]:
        t_block_mask = (data_label[:,0]>=start[0]) & (data_label[:,0]<(start[0]+length)) & \
                 (data_label[:,1]>=start[1]) & (data_label[:,1]<(start[1]+length)) & \
                 (data_label[:,2]>=start[2]) & (data_label[:,2]<(start[2]+length))
        t_block_data_label = data_label[t_block_mask]
        t_block_data_label = sample_data(t_block_data_label, npoints)
        t_block_data_label = np.expand_dims(t_block_data_label, axis=0)
        block_data_label = np.concatenate((block_data_label, t_block_data_label), axis=0)  # shape=(B,npoints,8)

    new_block_data_label = np.zeros((block_data_label.shape[0], npoints, 7+4))  # data+label

    min_block_zyx = np.min(block_data_label[:,:,0:3], axis=1)  # shape = (B,3)
    min_block_zyx = np.expand_dims(min_block_zyx, axis=1)  # shape = (B,1,3)
    new_block_data_label[:,:,0:3] = block_data_label[:,:,0:3]-(min_block_zyx+length/2.0)  # local geometry

    new_block_data_label[:,:,3] = block_data_label[:,:,3]/255.0  # ct value

    max_zyx = np.max(data_label[:,0:3], axis=0)  # shape = (3,)
    new_block_data_label[:,:,4:7] = block_data_label[:,:,0:3]/max_zyx   # global geometry
    new_block_data_label[:,:,7:] = block_data_label[:,:,4:]  # labels


    if flag_save:
        save_block_data_label = np.zeros(block_data_label.shape)
        save_block_data_label[:,:,0] = new_block_data_label[:,:,6]
        save_block_data_label[:,:,1] = new_block_data_label[:,:,5]
        save_block_data_label[:,:,2] = new_block_data_label[:,:,4]
        save_block_data_label[:,:,3] = new_block_data_label[:,:,3]
        save_block_data_label[:,:,4:] = new_block_data_label[:,:,7:]
        sub_block_folder_path = out_folder_path + '/' + ct_index
        if not os.path.exists(sub_block_folder_path):
            os.makedirs(sub_block_folder_path)

        for i_block in range(save_block_data_label.shape[0]):
            file_path = sub_block_folder_path + '/' + str(i_block) + '.txt'
            np.savetxt(file_path, save_block_data_label[i_block,:,:])

    return new_block_data_label


def simple_start(start, data_label, npoints, thickness):
    t_block_mask = (data_label[:, 0] >= start) & (data_label[:, 0] < (start + thickness))
    t_block_data_label = data_label[t_block_mask]
    t_block_data_label = sample_data(t_block_data_label, npoints)
    t_block_data_label = np.expand_dims(t_block_data_label, axis=0)
    return t_block_data_label   # shape = (npoints, 8)


def create_block_data_label2_mul(data_label, thickness, npoints, out_folder_path, ct_index, flag_save=0):
    # data_label, shape=(n,8), z,y,x,ct,l1,l2,l3,l4

    # normalize coordinates to original points
    min_zyx = np.min(data_label[:,0:3], axis=0)
    data_label[:,0:3] -= min_zyx

    max_z = np.max(data_label[:,0])
    starts = []
    num_starts = int(max_z/thickness)
    for i_start in range(num_starts):
        starts.append(i_start*thickness)
    if (max_z - num_starts*thickness)>0.1:
        starts.append(max_z-thickness)

    block_mask = (data_label[:,0]>=starts[0]) & (data_label[:,0]<(starts[0]+thickness))
    block_data_label = data_label[block_mask]
    block_data_label = sample_data(block_data_label, npoints)
    block_data_label = np.expand_dims(block_data_label, axis=0)  # shape=(1,npoints,8)

    results = []
    p = Pool(12)
    for start in starts[1:]:
        results.append(p.apply_async(simple_start, args=(start, data_label, npoints, thickness,)))

    p.close()
    p.join()

    block_data_label_list = []
    for res in results:
        block_data_label_list.append(res.get())

    block_data_label_t = np.concatenate(block_data_label_list, axis=0)
    block_data_label = np.concatenate((block_data_label, block_data_label_t), axis=0)

    max_value = np.max(data_label[:,0:4], axis=0)  # shape = (4, )
    block_data_label[:,:,0:4] /= max_value

    if flag_save:
        save_block_data_label = np.zeros(block_data_label.shape)
        save_block_data_label[:,:,0] = block_data_label[:,:,2]
        save_block_data_label[:,:,1] = block_data_label[:,:,1]
        save_block_data_label[:,:,2] = block_data_label[:,:,0]
        save_block_data_label[:,:,3:] = block_data_label[:,:,3:]
        sub_block_folder_path = out_folder_path + '/' + ct_index
        if not os.path.exists(sub_block_folder_path):
            os.makedirs(sub_block_folder_path)

        for i_block in range(save_block_data_label.shape[0]):
            file_path = sub_block_folder_path + '/' + str(i_block) + '.txt'
            np.savetxt(file_path, save_block_data_label[i_block,:,:])

    return block_data_label



def create_block_data_label2(data_label, thickness, npoints, out_folder_path, ct_index, flag_save=0):
    # data_label, shape=(n,8), z,y,x,ct,l1,l2,l3,l4

    # normalize coordinates to original points
    min_zyx = np.min(data_label[:,0:3], axis=0)
    data_label[:,0:3] -= min_zyx

    max_z = np.max(data_label[:,0])
    starts = []
    num_starts = int(max_z/thickness)
    for i_start in range(num_starts):
        starts.append(i_start*thickness)
    if (max_z - num_starts*thickness)>0.1:
        starts.append(max_z-thickness)

    block_mask = (data_label[:,0]>=starts[0]) & (data_label[:,0]<(starts[0]+thickness))
    block_data_label = data_label[block_mask]
    block_data_label = sample_data(block_data_label, npoints)
    block_data_label = np.expand_dims(block_data_label, axis=0) # shape=(1,npoints,8)

    for start in starts[1:]:
        t_block_mask = (data_label[:,0]>=start) & (data_label[:,0]<(start+thickness))
        t_block_data_label = data_label[t_block_mask]
        t_block_data_label = sample_data(t_block_data_label, npoints)
        t_block_data_label = np.expand_dims(t_block_data_label, axis=0)
        block_data_label = np.concatenate((block_data_label, t_block_data_label), axis=0)


    max_value = np.max(data_label[:,0:4], axis=0)  # shape = (4, )
    block_data_label[:,:,0:4] /= max_value

    if flag_save:
        save_block_data_label = np.zeros(block_data_label.shape)
        save_block_data_label[:,:,0] = block_data_label[:,:,2]
        save_block_data_label[:,:,1] = block_data_label[:,:,1]
        save_block_data_label[:,:,2] = block_data_label[:,:,0]
        save_block_data_label[:,:,3:] = block_data_label[:,:,3:]
        sub_block_folder_path = out_folder_path + '/' + ct_index
        if not os.path.exists(sub_block_folder_path):
            os.makedirs(sub_block_folder_path)

        for i_block in range(save_block_data_label.shape[0]):
            file_path = sub_block_folder_path + '/' + str(i_block) + '.txt'
            np.savetxt(file_path, save_block_data_label[i_block,:,:])

    return block_data_label


def create_block_data_label(data_label, thickness, npoints, out_folder_path, ct_index, flag_save=0):
    # data_label, shape=(n,8), z,y,x,ct,l1,l2,l3,l4

    # normalize coordinates to original points
    min_zyx = np.min(data_label[:,0:3], axis=0)
    data_label[:,0:3] -= min_zyx

    max_z = np.max(data_label[:,0])
    starts = []
    num_starts = int(max_z/thickness)
    for i_start in range(num_starts):
        starts.append(i_start*thickness)
    if (max_z - num_starts*thickness)>0.1:
        starts.append(max_z-thickness)

    block_mask = (data_label[:,0]>=starts[0]) & (data_label[:,0]<(starts[0]+thickness))
    block_data_label = data_label[block_mask]
    block_data_label = sample_data(block_data_label, npoints)
    block_data_label = np.expand_dims(block_data_label, axis=0) # shape=(1,npoints,8)

    for start in starts[1:]:
        t_block_mask = (data_label[:,0]>=start) & (data_label[:,0]<(start+thickness))
        t_block_data_label = data_label[t_block_mask]
        t_block_data_label = sample_data(t_block_data_label, npoints)
        t_block_data_label = np.expand_dims(t_block_data_label, axis=0)
        block_data_label = np.concatenate((block_data_label, t_block_data_label), axis=0)

    if flag_save:
        save_block_data_label = np.zeros(block_data_label.shape)
        save_block_data_label[:,:,0] = block_data_label[:,:,2]
        save_block_data_label[:,:,1] = block_data_label[:,:,1]
        save_block_data_label[:,:,2] = block_data_label[:,:,0]
        save_block_data_label[:,:,3:] = block_data_label[:,:,3:]
        sub_block_folder_path = out_folder_path + '/' + ct_index
        if not os.path.exists(sub_block_folder_path):
            os.makedirs(sub_block_folder_path)

        for i_block in range(save_block_data_label.shape[0]):
            file_path = sub_block_folder_path + '/' + str(i_block) + '.txt'
            np.savetxt(file_path, save_block_data_label[i_block,:,:])

    return block_data_label


def Guassian3d(distance):
    # distance, shape=(z, y, x, 3)
    sigma = 5.0/3.0
    t_square = np.square(distance)
    temp = -((t_square[:,:,:,0]+t_square[:,:,:,1]+t_square[:,:,:,2])/(2*pow(sigma,2)))
    guassian = np.exp(temp)
    return guassian


def add_label(voxel, ann, label_dict, Z_Y_X):
    # voxel, shape==(x,y,z,num_label+1), channel0 is ct value
    # ann, (voxel), z, y, x, dz, dy, dx, label
    # Z_Y_X, shape==(z,y,x,3), values are coordinates
    cz = ann[0]
    cy = ann[1]
    cx = ann[2]
    dz = ann[3]
    dy = ann[4]
    dx = ann[5]
    label = str(int(ann[6]))
    label = label_dict[label]

    d_max = np.max([dz, dy, dx])
    r_max = min(5.0, d_max/2.0)

    z_min = int(cz - r_max)
    z_min = max(0, z_min)
    z_max = int(np.ceil(cz + r_max))
    z_max = min(voxel.shape[0]-1, z_max)
    y_min = int(cy - r_max)
    y_min = max(0, y_min)
    y_max = int(np.ceil(cy + r_max))
    y_max = min(voxel.shape[1]-1, y_max)
    x_min = int(cx - r_max)
    x_min = max(0, x_min)
    x_max = int(np.ceil(cx + r_max))
    x_max = min(voxel.shape[2]-1, x_max)

    center = np.array([cz, cy, cx])
    distance = Z_Y_X[z_min:z_max, y_min:y_max, x_min:x_max] - center
    voxel[z_min:z_max, y_min:y_max, x_min:x_max, label] = Guassian3d(distance)

    return voxel


def create_block_label(ct_folder_path,ct_index,out_folder_path,one_annotations, save_flag=0, thickness=50.0, npoints=125000):
    # annotations: [[x,y,z,dx,dy,dz,label], ...], world
    air_threshold = -1.0
    label_dict = {'1': 1, '5': 2, '31': 3, '32': 4}
    num_label = len(label_dict.keys())
    out_points_folder_path = out_folder_path + '/points'
    out_blocks_folder_path = out_folder_path + '/blocks'
    if not os.path.exists(out_points_folder_path):
        os.makedirs(out_points_folder_path)
    if not os.path.exists(out_blocks_folder_path):
        os.makedirs(out_blocks_folder_path)


    ct_file_path = ct_folder_path + '/' + ct_index + '.mhd'
    itkimage = sitk.ReadImage(ct_file_path)
    itkimage_resampled = resample_image(itkimage)
    voxel_image = sitk.GetArrayFromImage(itkimage_resampled)
    voxel_image = windows(voxel_image)
    origin_re_w = np.array(list(reversed(itkimage_resampled.GetOrigin())))  # xyz -> zyx
    spacing_re_w = np.array(list(reversed(itkimage_resampled.GetSpacing())))  # xyz -> zyx

    # change annotation sequence into 'z, y, x, dz, dy, dx, label', (world)
    one_annotations = np.array(one_annotations)

    new_one_annotations = np.zeros(one_annotations.shape)
    new_one_annotations[:, 0] = one_annotations[:, 2]
    new_one_annotations[:, 1] = one_annotations[:, 1]
    new_one_annotations[:, 2] = one_annotations[:, 0]
    new_one_annotations[:, 3] = one_annotations[:, 5]
    new_one_annotations[:, 4] = one_annotations[:, 4]
    new_one_annotations[:, 5] = one_annotations[:, 3]
    new_one_annotations[:, 6] = one_annotations[:, 6]  # (world), z, y, x, dz, dy, dx, label

    # world to voxel
    new_one_annotations[:, 0:3] -= origin_re_w
    new_one_annotations[:, 0:3] /= spacing_re_w
    new_one_annotations[:, 3:6] /= spacing_re_w  # (voxel), z, y, x, dz, dy, dx, label

    v_shape = voxel_image.shape
    new_voxel_image = np.zeros((v_shape[0], v_shape[1], v_shape[2], num_label + 1))  # initialize points data, (z,y,x,5)
    new_voxel_image[:, :, :, 0] = voxel_image

    # voxel that contains coordinates as value
    zs = range(0, v_shape[0], 1)
    ys = range(0, v_shape[1], 1)
    xs = range(0, v_shape[2], 1)
    Zs, Ys, Xs = np.meshgrid(zs, ys, xs)

    Zs = np.expand_dims(Zs.transpose(1, 0, 2), axis=-1)
    Ys = np.expand_dims(Ys.transpose(1, 0, 2), axis=-1)
    Xs = np.expand_dims(Xs.transpose(1, 0, 2), axis=-1)

    Z_Y = np.concatenate((Zs, Ys), axis=-1)
    Z_Y_X = np.concatenate((Z_Y, Xs), axis=-1)  # shape=(z,y,x,3)  # voxel

    # coordinates from voxel to world
    Z_Y_X_world = Z_Y_X * spacing_re_w + origin_re_w

    for ann in new_one_annotations:
        new_voxel_image = add_label(new_voxel_image, ann, label_dict, Z_Y_X)

    new_voxel_image[v_shape[0] - 4:v_shape[0], :, :, 0] = 0   # delete useless plate

    voxel_data_label = np.concatenate((Z_Y_X_world, new_voxel_image), axis=-1)  # shape=(x,y,z,3+5), z,y,x,ct,l1,l2,l3,l4

    mask = new_voxel_image[:,:,:,0] > air_threshold
    data_label = voxel_data_label[mask]  # shape=(n,8)

    # save temporary points
    if save_flag:
        data_label_save = np.zeros(data_label.shape)
        data_label_save[:,0] = data_label[:,2]
        data_label_save[:,1] = data_label[:,1]
        data_label_save[:,2] = data_label[:,0]
        data_label_save[:,3:] = data_label[:,3:]
        out_points_path = out_points_folder_path + '/' + ct_index + '_points.txt'
        np.savetxt(out_points_path, data_label_save, fmt='%.8f')

    # create blocks
    block_data_label = create_block_data_label5(data_label, thickness, npoints, out_blocks_folder_path, ct_index, flag_save=save_flag)

    return block_data_label


#**************************test_main*********************
#
import time
ct_folder_path = '/media/jp/disk/dataset/lung/dataset/train'
out_folder_path = '/home/jp/point_cloud/temp'
if not os.path.exists(out_folder_path):
    os.makedirs(out_folder_path)
ct_index = '641414'
annotations_path = '/media/jp/disk/dataset/lung/dataset/chestCT_round1_annotation.csv'
start_time = time.clock()

annotations_dict = load_annotations(annotations_path)

block_data_label = create_block_label(ct_folder_path,ct_index,out_folder_path,annotations_dict[ct_index], save_flag=1)
print(block_data_label.shape)
print(block_data_label[0,0:10,:])
print('total time: %f' % (time.clock()-start_time))

