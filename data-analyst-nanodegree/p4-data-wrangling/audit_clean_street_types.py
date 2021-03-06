#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "sample.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", 
            "Square", "Lane", "Road", "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE

mapping = { "St": "Street",
            "St.": "Street",
            "Rd.": "Road",
            "Ave" : "Avenue",
            "Marg" : "Road",
            "Path" : "Road"
            }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r", encoding="utf8")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


def update_name(name, mapping):
    ''' update street names '''  
    if name.split()[-1] in mapping.keys():
        ChangedWord = mapping[name.split()[-1]]    
        name = name.split()[:-1]   
        name.append(ChangedWord)    
        name = " ".join(name)
    return name

def main():    
    st_types = audit(OSMFILE)    
    pprint.pprint(dict(st_types))        
    for st_type, ways in st_types.items():        
        if st_type in mapping:    
            for name in ways:
                better_name = update_name(name, mapping)                
                print (name, "=>", better_name)

if __name__ == "__main__":
   main()


