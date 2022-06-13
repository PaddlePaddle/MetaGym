#
# Generate Variant Ants with different lengths of legs
#

from gym import utils
import numpy
from numpy import random
import xml.etree.ElementTree as ET

Pattern_Clip_Min = [1.0, 0.1, 0.1] * 4
Pattern_Clip_Max = [2.0, 2.0, 2.0] * 4

def gen_pattern_tra():
    x = random.random(size=(12,)) + 0.5
    return numpy.clip(x, Pattern_Clip_Min, Pattern_Clip_Max)

def gen_pattern_tst():
    x = random.random(size=(12,)) + 0.5
    return numpy.clip(x, Pattern_Clip_Min, Pattern_Clip_Max)

def gen_pattern_ood():
    x = random.random(size=(12,)) + 0.5
    x = x - (x < 0.5) * 0.3 + (x > 0.5) * 0.3
    return numpy.clip(x, Pattern_Clip_Min, Pattern_Clip_Max)

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
    for i in range(64):
        reconfig_xml("ant.xml", gen_pattern_tra(), "ant_var_tra_%03d.xml"%i)
    for i in range(32):
        reconfig_xml("ant.xml", gen_pattern_tst(), "ant_var_tst_%03d.xml"%i)
    for i in range(32):
        reconfig_xml("ant.xml", gen_pattern_ood(), "ant_var_ood_%03d.xml"%i)
