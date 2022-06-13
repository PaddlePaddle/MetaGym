#
# Generate Variant Ants with different lengths of legs
#

from gym import utils
import numpy
from numpy import random
import xml.etree.ElementTree as ET

Pattern_Clip_Min = numpy.asarray([0.80, 0.60, 0.60])
Pattern_Clip_Max = numpy.asarray([1.20, 1.40, 1.40])

def gen_pattern_tra():
    x = random.random(size=(3,))
    return Pattern_Clip_Min + x * (Pattern_Clip_Max - Pattern_Clip_Min)

def gen_pattern_tst():
    x = random.random(size=(3,))
    return Pattern_Clip_Min + x * (Pattern_Clip_Max - Pattern_Clip_Min)

def gen_pattern_ood():
    x = random.random(size=(3,))
    return Pattern_Clip_Min + x * (Pattern_Clip_Max - Pattern_Clip_Min)

def reconfig_xml(file_name, pattern, output_file):
    tree = ET.parse(file_name)
    root = tree.getroot()
    wbody = root.find('worldbody')
    torso = wbody.find('body')
    pelvis = torso.find('body').find('body')
    pelv_l = numpy.asarray(list(map(float, pelvis.find("geom").get("fromto").split())))
    pelv_l[1] *= pattern[0]
    pelv_l[4] *= pattern[0]
    pelvis.find("geom").set("fromto", " ".join(map(str, pelv_l)))

    deta_h = 0.403 * (pattern[1] - 1.0) + 0.45 * (pattern[2] - 1.0) - 0.20

    pos = numpy.asarray(list(map(float, torso.get("pos").split())))
    pos[2] += deta_h
    torso.set("pos", " ".join(map(str, pos)))

    for thigh_body in pelvis.findall('body'):
        thigh_geom = thigh_body.find("geom")
        thin_body = thigh_body.find("body")
        thin_geom = thin_body.find("geom")
        foot = thin_body.find("body")

        pos = numpy.asarray(list(map(float, thigh_body.get("pos").split())))
        pos[1] *= pattern[0]
        thigh_body.set("pos", " ".join(map(str, pos)))

        l = numpy.asarray(list(map(float, thigh_geom.get("fromto").split())))
        l *= pattern[1]
        thigh_geom.set("fromto", " ".join(map(str, l)))
        pos = numpy.asarray(list(map(float, thin_body.get("pos").split())))
        pos *= pattern[1]
        thin_body.set("pos", " ".join(map(str, pos)))

        l = numpy.asarray(list(map(float, thin_geom.get("fromto").split())))
        l *= pattern[2]
        thin_geom.set("fromto", " ".join(map(str, l)))
        pos = numpy.asarray(list(map(float, foot.get("pos").split())))
        pos *= pattern[2]
        foot.set("pos", " ".join(map(str, pos)))

    f_out = open(output_file, "w")
    f_out.write(ET.tostring(root, encoding='unicode'))
    f_out.close()

if __name__=="__main__":
    for i in range(64):
        reconfig_xml("humanoid.xml", gen_pattern_tra(), "humanoid_var_tra_%03d.xml"%i)
    for i in range(32):
        reconfig_xml("humanoid.xml", gen_pattern_tst(), "humanoid_var_tst_%03d.xml"%i)
    for i in range(32):
        reconfig_xml("humanoid.xml", gen_pattern_ood(), "humanoid_var_ood_%03d.xml"%i)
