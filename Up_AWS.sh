Data_ID=$1
AWS_S3_PATH=s3://research.luffingfuturellc/Pangolin
aws s3 cp ./${Data_ID} $AWS_S3_PATH
