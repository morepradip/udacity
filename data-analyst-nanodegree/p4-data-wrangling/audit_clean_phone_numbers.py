#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET
from collections import defaultdict
import pprint

OSMFILE = "sample.osm" 

def audit_phone_type(phone_types, phone_number):
    ''' audit phone numbers'''
    
    if phone_number[0] == '0':
        phone_types['Phone_Starting_Zero'].add(phone_number)
    elif phone_number[0] == '+':
        phone_types['Phone_Starting_Plus'].add(phone_number)
    else:
        phone_types['Phone_Starting_Other'].add(phone_number)

def is_phone(elem):
    return (elem.attrib['k'] == "phone")

def audit(osmfile):
    osm_file = open(osmfile, "r", encoding="utf8")
    phone_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_phone(tag):
                    audit_phone_type(phone_types, tag.attrib['v'])
    osm_file.close()
    
    return phone_types


def update_phone_number(phone_number):
    ''' update phone numbers '''
    if phone_number[0] == '0':
        new_number = "+91" + phone_number[1:] 
        return new_number
    else: 
        return phone_number

def main():    
    phone_types = audit(OSMFILE)    
    pprint.pprint(dict(phone_types))        
    for phone_type, phone_numbers in phone_types.items():
        if phone_type == "Phone_Starting_Zero":
            for phone_number in phone_numbers:
                corrected_number = update_phone_number(phone_number)
                print(phone_number, "=>", corrected_number)


if __name__ == "__main__":
   main()


