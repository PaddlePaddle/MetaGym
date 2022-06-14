#
# Generate Variant Ants with different lengths of legs
#

from gym import utils
import numpy
from numpy import random
import xml.etree.ElementTree as ET

Pattern_Clip_Min = numpy.asarray([1.0, 0.60, 0.60] * 4)
Pattern_Clip_Max = numpy.asarray([1.50, 1.40, 1.40] * 4)

def gen_pattern_base():
    x = random.random(size=(12,))
    return Pattern_Clip_Min + x * (Pattern_Clip_Max - Pattern_Clip_Min)

def pattern_ood_clustering(pattern_list, top_k):
    patterns = numpy.array(pattern_list)
    center = numpy.mean(patterns, axis=0)
    deta = patterns - center
    dists = numpy.sqrt(numpy.sum(deta * deta, axis=-1))
    dist_idx = numpy.argsort(-dists)
    top_k_list = []
    residue = []
    for i, rank in enumerate(dist_idx):
        if(i < top_k):
            top_k_list.append(pattern_list[rank])
        else:
            residue.append(pattern_list[rank])

    return top_k_list, residue

def reconfig_xml(file_name, pattern, output_file):
    tree = ET.parse(file_name)
    root = tree.getroot()
    wbody = root.find('worldbody')
    body = wbody.find('body')
    idx = 0
    for subbody in body.findall('body'):
        subsubbody = subbody.find("body")
        subsubsubbody = subsubbody.find("body")
        l1 = numpy.asarray(list(map(float, subbody.find("geom").get("fromto").split())))
        l2 = numpy.asarray(list(map(float, subsubbody.find("geom").get("fromto").split())))
        l3 = numpy.asarray(list(map(float, subsubsubbody.find("geom").get("fromto").split())))
        l1 *= pattern[idx]
        l2 *= pattern[idx + 1]
        l3 *= pattern[idx + 2]
        idx += 3
        subbody.find("geom").set("fromto", " ".join(map(str, l1)))
        subbody.find("body").set("pos", " ".join(map(str, l1[3:])))
        subsubbody.find("geom").set("fromto", " ".join(map(str, l2)))
        subsubbody.find("body").set("pos", " ".join(map(str, l2[3:])))
        subsubsubbody.find("geom").set("fromto", " ".join(map(str, l3)))
    f_out = open(output_file, "w")
    f_out.write(ET.tostring(root, encoding='unicode'))
    f_out.close()

if __name__=="__main__":
    pattern_list = [gen_pattern_base() for _ in range(384)]
    ood_pattern, others = pattern_ood_clustering(pattern_list, 64) 
    
    for i, pattern in enumerate(ood_pattern):
        reconfig_xml("ant.xml", pattern, "ant_var_ood_%03d.xml"%i)
    for i, pattern in enumerate(others[:256]):
        reconfig_xml("ant.xml", pattern, "ant_var_tra_%03d.xml"%i)
    for i, pattern in enumerate(others[256:]):
        reconfig_xml("ant.xml", pattern, "ant_var_tst_%03d.xml"%i)
