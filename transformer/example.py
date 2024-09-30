#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:11:48 2024

@author: justin
"""

import pickle as pkl



## Initial loading of the dataset
train_data = pkl.load(open("train_data_sample.pkl", "rb"))

# train_data_sample = train_data[0]

# #save the first sample
#pkl.dump(train_data_sample, open("train_data_sample.pkl", "wb"))

observation = train_data['observation']
ego_vehicle_descriptor = observation['ego_vehicle_descriptor']
route_descriptor = observation['route_descriptors']

