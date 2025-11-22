import boto3
import pandas as pd
from io import StringIO
import logging
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class S3DataLoader:
    def __init__(self, bucket_name, aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), region_name=os.getenv("AWS_REGION")):
        self.bucket_name = bucket_name
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            self.s3_client = boto3.client('s3', region_name=region_name)

    def load_csv_from_s3(self, file_key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
        return pd.read_csv(obj['Body'])

    def list_files(self, prefix=''):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []