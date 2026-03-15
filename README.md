# ☁️ AWS SageMaker MLOps Pipeline

[![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Terraform](https://img.shields.io/badge/Terraform-1.8-7B42BC?style=flat-square&logo=terraform&logoColor=white)](https://terraform.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> End-to-end MLOps on AWS.— SageMaker Pipelines for training, Model Registry for versioning, real-time + batch inference, and Model Monitor for drift detection. IaC with Terraform.

## 🏗 Architecture
```
S3 (data) → SageMaker Pipeline → Model Registry → Endpoint (real-time / batch)
                                                       ↓
                                              Model Monitor → SNS Alert
```

## 📦 What's Included
- **SageMaker Pipeline**: preprocessing → training → evaluation → conditional registration
- **Model Registry**: automatic approval workflow with approval gates
- **Inference**: real-time endpoint + async endpoint + batch transform
- **Model Monitor**: data quality, model quality, bias, and explainability monitors
- **Terraform**: full IaC for all SageMaker resources

## 🚀 Deploy

```bash
git clone https://github.com/harshadkhetpal/aws-sagemaker-mlops
cd aws-sagemaker-mlops

# Deploy infrastructure
cd terraform && terraform init && terraform apply

# Run training pipeline
python pipelines/run_pipeline.py \
  --pipeline-name HarshadMLPipeline \
  --role-arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
  --s3-bucket s3://harshad-ml-bucket
```
