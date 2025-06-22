# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 08:23:44 2025

@author: joseph
"""

import boto3
from botocore.exceptions import NoCredentialsError

# Upload file to S3 bucket
def upload_to_s3(file_name, bucket, object_name=None):
    s3_client = boto3.client('s3')
    if object_name is None:
        object_name = file_name
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"Uploaded {file_name} to s3://{bucket}/{object_name}")
    except NoCredentialsError:
        print("AWS credentials not found.")
        raise

# Download file from S3 bucket
def download_from_s3(bucket, object_name, download_path):
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket, object_name, download_path)
        print(f"Downloaded s3://{bucket}/{object_name} to {download_path}")
    except NoCredentialsError:
        print("AWS credentials not found.")
        raise
