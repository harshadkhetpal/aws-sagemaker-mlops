# Author: Harshad Khetpal <harshadkhetpal@gmail.com>
# SageMaker Model Monitor — Drift Detection Setup

import boto3, sagemaker
from sagemaker.model_monitor import (
    DataCaptureConfig, DefaultModelMonitor,
    ModelQualityMonitor, CronExpressionGenerator
)


def setup_data_capture(endpoint_name: str, s3_capture_path: str, role: str):
    """Enable data capture on a SageMaker real-time endpoint."""
    sm = boto3.client("sagemaker")
    sm.update_endpoint(
        EndpointName=endpoint_name,
        RetainAllVariantProperties=True,
        DeploymentConfig={
            "DataCaptureConfig": {
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": s3_capture_path,
                "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            }
        },
    )


def create_data_quality_monitor(endpoint_name: str, baseline_s3: str, output_s3: str, role: str):
    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )
    monitor.suggest_baseline(
        baseline_dataset=baseline_s3,
        dataset_format=sagemaker.model_monitor.DatasetFormat.csv(header=True),
        output_s3_uri=f"{output_s3}/baseline",
        wait=True,
    )
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{endpoint_name}-data-quality",
        endpoint_input=endpoint_name,
        output_s3_uri=f"{output_s3}/reports",
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator(hourly(),
        enable_cloudwatch_metrics=True,
    )
    print(f"✅ Data quality monitor created for {endpoint_name}")
