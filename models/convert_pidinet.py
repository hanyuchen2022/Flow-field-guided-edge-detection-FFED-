
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import config_model_converted

def convert_pdc(op, weight):
    if op == 'cv':
        return weight
    elif op == 'cd':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
        return weight

    elif op == 'ad':
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight_conv = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
        return weight_conv
    elif op == 'rd':
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return buffer
    elif op == 'als':
        shape = weight.shape
        buffer1 = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        buffer2 = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        buffer_sum = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer1[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer1[:, :, [6, 7, 8, 11, 14, 16, 17, 18]] = weight[:, :, 1:]
        buffer2[:, :, [10, 0, 2,20, 4, 22, 24, 14]] = -weight[:, :, 1:]
        buffer2[:, :, [11, 6, 7, 16, 8, 17, 18, 13]] = -weight[:, :, 1:]

        buffer_sum = buffer1 + buffer2
        buffer_sum[:, :, 12] = 0
        buffer = buffer_sum.view(shape[0], shape[1], 5, 5)
        return buffer
    raise ValueError("wrong op {}".format(str(op)))

def convert_ffed(state_dict, config):
    pdcs = config_model_converted(config)
    new_dict = {}
    for pname, p in state_dict.items():
        if 'init_block.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[0], p)
        elif 'block1_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[1], p)
        elif 'block1_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[2], p)
        elif 'block1_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[3], p)
        elif 'block2_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[4], p)
        elif 'block2_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[5], p)
        elif 'block2_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[6], p)
        elif 'block2_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[7], p)
        elif 'block3_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[8], p)
        elif 'block3_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[9], p)
        elif 'block3_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[10], p)
        elif 'block3_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[11], p)
        elif 'block4_1.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[12], p)
        elif 'block4_2.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[13], p)
        elif 'block4_3.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[14], p)
        elif 'block4_4.conv1.weight' in pname:
            new_dict[pname] = convert_pdc(pdcs[15], p)
        else:
            new_dict[pname] = p

    return new_dict

