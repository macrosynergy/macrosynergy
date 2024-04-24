import boto3
import os
import re
import time
from botocore.exceptions import ClientError
import pandas as pd

AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
REGION_NAME = os.getenv('REGION_NAME')
ECR_IMAGE = os.getenv('ECR_IMAGE')
ECS_CLUSTER_NAME = os.getenv('ECS_CLUSTER_NAME')
SUBNET_IDS = os.getenv('SUBNET_IDS').split(' ')
EXECUTION_ROLE_ARN = os.getenv('EXECUTION_ROLE_ARN')

start_time = time.time()


def clean_string(s):
    return re.sub(r"[^a-zA-Z0-9\-_]", "", s)


def get_notebooks(bucket_name):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    notebooks = [obj.key for obj in bucket.objects.all() if obj.key.endswith(".ipynb")]

    notebooks.remove("Regression-based_macro_trading_signals.ipynb") # Requires 64 GB of memory currently which is too expensive

    return notebooks


def run_task(
    ecr_image,
    ecs_client,
    nb_name,
    bucket_name,
    log_group_name,
    region_name,
    ecs_cluster_name,
    subnet_ids,
    cpu,
    memory,
    execution_role_arn,
):
    task_def_response = ecs_client.register_task_definition(
        family=clean_string(nb_name.replace(".ipynb", "")),
        containerDefinitions=[
            {
                "name": clean_string(nb_name.replace(".ipynb", "")),
                "image": ecr_image,
                "cpu": cpu,
                "memory": memory,
                "essential": True,
                "environment": [
                    {
                        "name": "NOTEBOOK_URL",
                        "value": "https://"
                        + bucket_name
                        + ".s3."
                        + region_name
                        + ".amazonaws.com/"
                        + nb_name,
                    },
                    {
                        "name": "BRANCH_NAME",
                        "value": "test",
                    },
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group_name,
                        "awslogs-region": region_name,
                        "awslogs-stream-prefix": "ecs",
                    },
                },
            },
        ],
        requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc",
        memory=str(memory),
        cpu=str(cpu),
        executionRoleArn=execution_role_arn,
        taskRoleArn=execution_role_arn,
    )

    task_definition = task_def_response["taskDefinition"]["taskDefinitionArn"]

    response = ecs_client.run_task(
        cluster=ecs_cluster_name,
        launchType="FARGATE",
        taskDefinition=task_definition,
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {"subnets": subnet_ids, "assignPublicIp": "ENABLED"}
        },
    )

    task_arn = response["tasks"][0]["taskArn"]
    return task_arn

notebooks = get_notebooks(AWS_BUCKET_NAME)
ecs_client = boto3.client("ecs", region_name=REGION_NAME)

task_arns = []

for notebook in notebooks:
    if notebook in [
        "Signal_optimization_basics.ipynb",
        #"Regression-based_macro_trading_signals.ipynb",
        "Regression-based_FX_signals.ipynb",
    ]:
        cpu = 16384
        memory = 32768
    else:
        cpu = 4096
        memory = 16384
    task_arns.append(
        run_task(
            ecr_image=ECR_IMAGE,
            ecs_client=ecs_client,
            nb_name=notebook,
            bucket_name=AWS_BUCKET_NAME,
            log_group_name="/ecs/",
            region_name=REGION_NAME,
            ecs_cluster_name=ECS_CLUSTER_NAME,
            subnet_ids=SUBNET_IDS,
            cpu=cpu,
            memory=memory,
            execution_role_arn=EXECUTION_ROLE_ARN,
        )
    )

print("ALL TASKS ARE RUNNING!")

nb_exit_codes = {"succeeded": [], "failed": []}

while len(task_arns) > 0:
    for task_arn in task_arns:
        response = ecs_client.describe_tasks(cluster=ECS_CLUSTER_NAME, tasks=[task_arn])
        task = response["tasks"][0]

        if task["lastStatus"] == "STOPPED":
            exit_code = task["containers"][0]["exitCode"]
            task_arns.remove(task_arn)
            # Get task name
            nb_name = task["containers"][0]["name"]
            if exit_code == 0:
                print(f"{nb_name} succeeded!")
                nb_exit_codes["succeeded"].append(nb_name)
            else:
                print(f"{nb_name} failed!")
                nb_exit_codes["failed"].append(nb_name)


end_time = time.time()

total_time = end_time - start_time

print(total_time)