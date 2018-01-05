# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:24:01 2017

@author: 157741
"""
import pandas as pd

def convert_datatype(df):   
    
    ''' converts data types of different features to appropriate data types '''
    
    #financial features data conversion
    df['bonus'] = df['bonus'].astype('float64')
    df['deferral_payments'] = df['deferral_payments'].astype('float64')
    df['deferred_income'] = df['deferred_income'].astype('float64')
    df['director_fees'] = df['director_fees'].astype('float64')
    df['salary'] = df['salary'].astype('float64')
    df['total_payments'] = df['total_payments'].astype('float64')
    df['loan_advances'] = df['loan_advances'].astype('float64')
    df['restricted_stock_deferred'] = df['restricted_stock_deferred'].astype('float64')
    df['total_stock_value'] = df['total_stock_value'].astype('float64')
    df['expenses'] = df['expenses'].astype('float64')
    df['exercised_stock_options'] = df['exercised_stock_options'].astype('float64')
    df['other'] = df['other'].astype('float64')
    df['long_term_incentive'] = df['long_term_incentive'].astype('float64')
    df['restricted_stock'] = df['restricted_stock'].astype('float64')
    
    # label feature
    df['poi'] = df['poi'].astype('bool')
    
    # Email features data conversion
    df['email_address'] = df['email_address'].astype('object') 
    df['to_messages'] = pd.to_numeric(df['to_messages'], errors='coerce')
    df['from_messages'] = pd.to_numeric(df['from_messages'], errors='coerce')
    df['from_poi_to_this_person'] = pd.to_numeric(df['from_poi_to_this_person'], errors='coerce')
    df['from_this_person_to_poi'] = pd.to_numeric(df['from_this_person_to_poi'], errors='coerce')
    df['shared_receipt_with_poi'] = pd.to_numeric(df['shared_receipt_with_poi'], errors='coerce')

    return df

if __name__ == '__main__':
    convert_datatype()