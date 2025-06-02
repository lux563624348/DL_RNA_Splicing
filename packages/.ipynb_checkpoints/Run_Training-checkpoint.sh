Data_ID=$1
AWS_S3_PATH=s3://research.luffingfuturellc/Pangolin
aws s3 cp $AWS_S3_PATH/${Data_ID} ./${Data_ID}

python ./packages/training.py --input ./${Data_ID} --epochs 30 --model exp --output ${Data_ID/.pt/_model.pt}

aws s3 cp ${Data_ID/.pt/_model.pt} $AWS_S3_PATH