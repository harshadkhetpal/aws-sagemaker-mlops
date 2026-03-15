# Author: Harshad Khetpal <harshadkhetpal@gmail.com>
# AWS SageMaker MLOps — Training Pipeline Definition

import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
import sagemaker


def build_pipeline(role: str, bucket: str, region: str = "us-east-1") -> Pipeline:
    sess = sagemaker.Session(boto3.Session(region_name=region))

    # ── Parameters ──────────────────────────────────────────────────────────
    model_approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
    accuracy_threshold = ParameterFloat("AccuracyThreshold", default_value=0.85)
    input_data = ParameterString("InputData", default_value=f"s3://{bucket}/data/raw")

    # ── Step 1: Preprocessing ────────────────────────────────────────────────
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=sess,
    )
    preprocessing_step = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[{"input_name": "raw", "source": input_data, "destination": "/opt/ml/processing/input"}],
        outputs=[{"output_name": "train", "source": "/opt/ml/processing/train", "destination": f"s3://{bucket}/processed/train"},
                 {"output_name": "test",  "source": "/opt/ml/processing/test",  "destination": f"s3://{bucket}/processed/test"}],
        code="training/preprocess.py",
    )

    # ── Step 2: Training ─────────────────────────────────────────────────────
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1"),
        role=role,
        instance_count=1,
        instance_type="ml.m5.2xlarge",
        output_path=f"s3://{bucket}/models",
        sagemaker_session=sess,
        hyperparameters={"max_depth": 6, "eta": 0.2, "num_round": 100, "objective": "binary:logistic"},
    )
    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={"train": sagemaker.TrainingInput(preprocessing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri)},
    )

    # ── Step 3: Evaluation ───────────────────────────────────────────────────
    eval_processor = SKLearnProcessor(framework_version="1.2-1", role=role,
                                       instance_type="ml.m5.xlarge", instance_count=1)
    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[{"input_name": "model", "source": training_step.properties.ModelArtifacts.S3ModelArtifacts, "destination": "/opt/ml/processing/model"},
                {"input_name": "test",  "source": preprocessing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, "destination": "/opt/ml/processing/test"}],
        outputs=[{"output_name": "evaluation", "source": "/opt/ml/processing/evaluation", "destination": f"s3://{bucket}/evaluation"}],
        code="training/evaluate.py",
    )

    return Pipeline(
        name="HarshadMLPipeline",
        parameters=[model_approval, accuracy_threshold, input_data],
        steps=[preprocessing_step, training_step, eval_step],
        sagemaker_session=sess,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--role-arn", required=True)
    p.add_argument("--s3-bucket", required=True)
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()
    pipeline = build_pipeline(args.role_arn, args.s3_bucket, args.region)
    pipeline.upsert(role_arn=args.role_arn)
    execution = pipeline.start()
    print(f"Pipeline started: {execution.arn}")
    execution.wait()
