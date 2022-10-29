import boto3
import os
import argparse
from pathlib2 import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='fetching args'
    )

    parser.add_argument('-b', '--base_path',
    default='/scripts/')

    parser.add_argument('-m','--model', default='model')

    parser.add_argument('-s','--s3_bucket', default='s3://kubeflow-pipeline-data')

    args = parser.parse_args()

    modelloc = Path(args.base_path + args.model).resolve(strict=False)

    modelf = modelloc.joinpath('model.pt')

    s3 = boto3.resource('s3', region_name='xyz', 
                        aws_access_key_id='xyz',
                        aws_secret_access_key='xyz'
                        )
    
    s3.Bucket(args.s3_bucket).upload_file(
        Filename=str(modelf), Key='model.pt')