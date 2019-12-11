"""
Helper script to generate an example txt map config.
"""
import numpy as np


def generate_map(length, width, height, map_file):
    map_matrix = np.zeros([width, length], dtype=np.int)

    # bounding wall
    # map_matrix[0, :] = height
    # map_matrix[-1, :] = height
    # map_matrix[:, 0] = height
    # map_matrix[:, -1] = height

    # obstacle wall
    s1, s2 = width // 5, length // 5
    # map_matrix[s1*2:s1*3, s2] = height
    # map_matrix[s1, s2*2:s2*3] = height
    # map_matrix[s1*2:s1*3, s2*4] = height
    # map_matrix[s1*3, s2*3:s2*4] = height

    # quadrotor pos
    map_matrix[s1, s2] = -1

    pos_len = len(str(height))
    with open(map_file, 'w') as f:
        for i in range(width):
            line = ' '.join([str(pos).zfill(pos_len) for pos in
                             list(map_matrix[i, :])])
            f.write(line + '\n')


if __name__ == '__main__':
    generate_map(20, 20, 10, 'default_map.txt')
