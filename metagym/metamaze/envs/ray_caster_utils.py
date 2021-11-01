import numpy
import pygame
import time
from numba import njit
from metagym.metamaze.envs.dynamics import PI, PI_4

FAR_RGB = numpy.array([0, 0, 0], dtype="float32")
TRANSPARENT_RGB = numpy.array([0, 255, 0], dtype="float32")

@njit(cache=True)
def DDA_2D(pos, i, j, cell_number, cell_size, cos_ori, sin_ori, cell_walls, cell_transparent, max_vision):
    delta_dist_x = 1.0e+6 if abs(cos_ori) < 1.0e-6 else abs(cell_size / cos_ori)
    delta_dist_y = 1.0e+6 if abs(sin_ori) < 1.0e-6 else abs(cell_size / sin_ori)
    d_x = ((i + 1) * cell_size - pos[0]) if cos_ori > 0 else (i * cell_size - pos[0])
    d_y = ((j + 1) * cell_size - pos[1]) if sin_ori > 0 else (j * cell_size - pos[1])
    side_dist_x = 1.0e+6 if abs(cos_ori) < 1.0e-6 else d_x / cos_ori
    side_dist_y = 1.0e+6 if abs(sin_ori) < 1.0e-6 else d_y / sin_ori
    delta_i = 1 if(cos_ori > 0) else -1
    delta_j = 1 if(sin_ori > 0) else -1
    hit_i = i
    hit_j = j
    hit_dist = 0.0
    hit_side = 0
    hit_transparent_list = []
    if(cell_transparent[hit_i, hit_j] > 0):
        if(side_dist_x < side_dist_y):
            hit_transparent_list.append((side_dist_x, hit_i, hit_j, 0))
        else:
            hit_transparent_list.append((side_dist_y, hit_i, hit_j, 1))

    while hit_dist < max_vision:
        if(side_dist_x < side_dist_y):
            hit_i += delta_i
            side_dist_y -= side_dist_x
            hit_dist += side_dist_x
            if(hit_i < 0 or hit_i >= cell_number):
                if(hit_j < 0 or hit_j >= cell_number):
                    hit_dist = 1.0e+6
                    break
            else:
                if(cell_transparent[hit_i, hit_j] > 0):
                    hit_transparent_list.append((hit_dist, hit_i, hit_j, 0))
                if(cell_walls[hit_i, hit_j] > 0):
                    hit_side = 0
                    break
            side_dist_x = delta_dist_x
        else:
            hit_j += delta_j
            side_dist_x -= side_dist_y
            hit_dist += side_dist_y
            if(hit_i < 0 or hit_i >= cell_number):
                if(hit_j < 0 or hit_j >= cell_number):
                    hit_dist = 1.0e+6
                    break
            else:
                if(cell_transparent[hit_i, hit_j] > 0):
                    hit_transparent_list.append((hit_dist, hit_i, hit_j, 1))
                if(cell_walls[hit_i, hit_j] > 0):
                    hit_side = 1
                    break
            side_dist_y = delta_dist_y
    return hit_dist, hit_i, hit_j, hit_side, hit_transparent_list


@njit(cache=True)
def maze_view(pos, ori, vision_height, cell_walls, cell_transparent, cell_texts, cell_size, texture_array,
        ceil_text, ceil_height, text_size, max_vision, l_focal, vision_angle_h, resolution_h, resolution_v):
    vision_screen_half_size_h = numpy.tan(vision_angle_h / 2) * l_focal
    vision_screen_half_size_v = vision_screen_half_size_h * resolution_v / resolution_h
    pixel_size = 2.0 * vision_screen_half_size_h / resolution_h
    s_ori = numpy.sin(ori)
    c_ori = numpy.cos(ori)
    text_to_cell = text_size / cell_size
    max_cell_i = cell_walls.shape[0]
    max_cell_j = cell_walls.shape[1]
    pixel_factor = pixel_size / l_focal
    transparent_factor = 0.30

    # prepare some maths
    rgb_array = numpy.zeros(shape=(resolution_h, resolution_v, 3), dtype="int32")
    transparent_array = numpy.zeros(shape=(resolution_h, resolution_v), dtype="float32")
    rgb_array[:, :] = FAR_RGB
    cos_hp_array = numpy.zeros(shape=(resolution_h,), dtype="float32")
    cos_abs_hp_array = numpy.zeros(shape=(resolution_h,), dtype="float32")
    sin_abs_hp_array = numpy.zeros(shape=(resolution_h,), dtype="float32")
    tan_hp = (- 0.5 - resolution_h / 2) * pixel_factor
    for d_h in range(resolution_h):
        tan_hp += pixel_factor
        cos_hp = numpy.sqrt(1.0 / (1.0 + tan_hp ** 2))
        sin_hp = tan_hp * cos_hp
        sin_abs_hp_array[d_h] = sin_hp * c_ori + cos_hp * s_ori
        cos_abs_hp_array[d_h] = cos_hp * c_ori - sin_hp * s_ori
        cos_hp_array[d_h] = cos_hp
    
    # paint floor
    for d_v in range(resolution_v - 1, resolution_v//2, -1):
        # horizontal distance on screen
        v_screen = (d_v + 0.5) * pixel_size - vision_screen_half_size_v
        distance = vision_height / v_screen * l_focal
        light_incident = v_screen / l_focal
        if(distance > max_vision):
            continue

        for d_h in range(resolution_h):
            eff_distance = distance / cos_hp_array[d_h]
            alpha = min(1.0, max(2.0 * eff_distance / max_vision - 1.0, 0.0)) * light_incident
            hit_x = eff_distance * cos_abs_hp_array[d_h] + pos[0]
            hit_y = eff_distance * sin_abs_hp_array[d_h] + pos[1]
            i = (hit_x / cell_size)
            j = (hit_y / cell_size)
            d_i = i - numpy.floor(i)
            d_j = j - numpy.floor(j)
            i = int(i)
            j = int(j)
            if(i < max_cell_i and i >= 0 and j < max_cell_j and j >= 0):
                text_id = cell_texts[i,j]
                d_i /= text_to_cell
                d_j /= text_to_cell
                d_i -= numpy.floor(d_i)
                d_j -= numpy.floor(d_j)
                d_i *= texture_array[text_id].shape[0]
                d_j *= texture_array[text_id].shape[1]
                rgb_array[d_h, d_v, :] = light_incident * (alpha * FAR_RGB + (1.0 - alpha) * texture_array[text_id][int(d_i), int(d_j)])
                if(cell_transparent[i, j] > 0):
                    rgb_array[d_h, d_v] = (1.0 - transparent_factor) * rgb_array[d_h, d_v] + transparent_factor * TRANSPARENT_RGB
                    transparent_array[d_h, d_v] = 1.0

    # paint ceil
    for d_v in range(resolution_v//2):
        v_screen = vision_screen_half_size_v - (d_v + 0.5) * pixel_size
        distance = (ceil_height - vision_height) / v_screen * l_focal
        light_incident = v_screen / l_focal
        if(distance > max_vision):
            continue

        for d_h in range(resolution_h):
            eff_distance = distance / cos_hp_array[d_h]
            alpha = min(1.0, max(2.0 * eff_distance / max_vision - 1.0, 0.0))
            hit_x = eff_distance * cos_abs_hp_array[d_h] + pos[0]
            hit_y = eff_distance * sin_abs_hp_array[d_h] + pos[1]
            t_i = int(hit_x / cell_size)
            t_j = int(hit_y / cell_size)
            i = (hit_x / text_size)
            j = (hit_y / text_size)
            d_i = i - numpy.floor(i)
            d_j = j - numpy.floor(j)
            d_i *= ceil_text.shape[0]
            d_j *= ceil_text.shape[1]
            rgb_array[d_h, d_v, :] = light_incident * (alpha * FAR_RGB + (1.0 - alpha) * ceil_text[int(d_i), int(d_j)])
            if(t_i >= 0 and t_i < max_cell_i and t_j >= 0 and t_j < max_cell_j and cell_transparent[t_i, t_j] > 0):
                rgb_array[d_h, d_v] = (1.0 - transparent_factor) * rgb_array[d_h, d_v] + transparent_factor * TRANSPARENT_RGB
                transparent_array[d_h, d_v] = 1.0
    
    # paint wall
    for d_h in range(resolution_h):
        i = int(pos[0] / cell_size)
        j = int(pos[1] / cell_size)
        hit_dist, hit_i, hit_j, hit_side, hit_transparent = DDA_2D(
                pos, i, j, max_cell_i, cell_size, cos_abs_hp_array[d_h], sin_abs_hp_array[d_h], 
                cell_walls, cell_transparent, max_vision)
        if(hit_dist > max_vision):
            continue
        alpha = min(1.0, max(2.0 * hit_dist / max_vision - 1.0, 0.0))
        text_id = cell_texts[hit_i,hit_j]
        hit_pt_x = hit_dist * cos_abs_hp_array[d_h] + pos[0]
        hit_pt_y = hit_dist * sin_abs_hp_array[d_h] + pos[1]
        if(hit_side == 0): # hit vertical wall
           local_h = hit_pt_y / cell_size 
           local_h -= numpy.floor(local_h)
           light_incident = abs(cos_abs_hp_array[d_h])
        else:
           local_h = hit_pt_x / cell_size 
           local_h -= numpy.floor(local_h)
           light_incident = abs(sin_abs_hp_array[d_h])

        ratio = hit_dist * cos_hp_array[d_h] / l_focal
        top_v = (ceil_height - vision_height) / ratio
        bot_v = vision_height / ratio

        v_s = max(0, int((vision_screen_half_size_v - top_v) / pixel_size))
        v_e = min(resolution_v, int((vision_screen_half_size_v + bot_v) / pixel_size))

        for d_v in range(v_s, v_e):
            local_v = (vision_screen_half_size_v - (d_v + 0.5) * pixel_size) * ratio + vision_height
            d_i = local_h / text_size
            d_j = local_v / text_size
            d_i -= numpy.floor(d_i)
            d_j -= numpy.floor(d_j)
            d_i = int(texture_array[text_id].shape[0] * d_i)
            d_j = int(texture_array[text_id].shape[1] * d_j)
            rgb_array[d_h, d_v, :] = light_incident * (alpha * FAR_RGB + (1.0 - alpha) * texture_array[text_id][int(d_i), int(d_j)])

        # Add those transparent
        for hit_dist, hit_i, hit_j, hit_side in hit_transparent:
            ratio = hit_dist * cos_hp_array[d_h] / l_focal
            top_v = (ceil_height - vision_height) / ratio
            bot_v = vision_height / ratio

            v_s = max(0, int((vision_screen_half_size_v - top_v) / pixel_size))
            v_e = min(resolution_v, int((vision_screen_half_size_v + bot_v) / pixel_size))
            for d_v in range(v_s, v_e):
                if(transparent_array[d_h, d_v] < 1):
                    rgb_array[d_h, d_v] = (1.0 - transparent_factor) * rgb_array[d_h, d_v] + transparent_factor * TRANSPARENT_RGB

    transparent_array =  numpy.expand_dims(transparent_array, axis=2)
        
    return rgb_array
