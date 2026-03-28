import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
import os

def find_non_uploaded_files(bucket_name, s3_prefix, local_prefix):
    not_uploaded_files = []
    ob = client.list_objects(Bucket=bucket_name, Prefix=s3_prefix)
    for file in os.listdir(local_prefix):
        file_path = os.path.join(local_prefix, file)
        if os.path.isfile(file_path) and file.endswith(".zip"):
            s3_key = os.path.join(s3_prefix, file)
            if not any(key['Key'] == s3_key for key in ob.get('Contents', [])):
                print(f"File \033[1;32;40m '{file}' \033[0;37;40m is \033[1;31;40m NOT uploaded \033[0;37;40m to S3.")
                not_uploaded_files.append(file)
    return not_uploaded_files

def upload_files_to_s3(local_prefix, files, bucket_name, s3_key, config):
    for file in files:
        file_path = os.path.join(local_prefix, file)
        s3_key_path = os.path.join(s3_key, file)
        print(f"Uploading file \033[1;32;40m '{file_path}' \033[0;37;40m to bucket \033[1;34;40m '{bucket_name}' \033[0;37;40m with key \033[1;34;40m '{s3_key_path}' \033[0;37;40m...")
        client.upload_file(file_path, bucket_name, s3_key_path, Config=config)
        print("Upload completed.")


s3_url = "https://s3.epfl.ch"
rw_access_key = ""
rw_secret_key = ""
bucket_name = ""

client = boto3.client(
    's3',
    aws_access_key_id=rw_access_key,
    aws_secret_access_key=rw_secret_key,
    endpoint_url=s3_url,
    config=Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

transfer_config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,
    multipart_chunksize=100 * 1024 * 1024,
    max_concurrency=8,
    use_threads=True,
)

local_prefix = "/media/experiments/ns-allinone-3.41/ns-3.41/scratch/"
s3_prefix = "dir/to/upload"
not_uploaded_files = find_non_uploaded_files(bucket_name, s3_prefix, local_prefix)
upload_files_to_s3(local_prefix, not_uploaded_files, bucket_name, s3_prefix, transfer_config)


# ob = client.list_objects(Bucket=bucket_name, Prefix=s3_prefix)
# for key in ob['Contents']:
#     print(key['Key'])
#     # If you want the file directly
#     client.download_file(bucket_name, key['Key'], key["Key"])

#     # If you want to read the bytes directly (works with text-based files)
#     content_object = client.get_object(Bucket=bucket_name, Key=key["Key"])
#     file_content = content_object['Body'].read().decode('utf-8')

#     # Deletes the file
#     client.delete_object(Bucket=bucket_name, Key=key["Key"])