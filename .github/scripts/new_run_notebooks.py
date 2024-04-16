import boto3
import uuid

# AWS Configurations
REGION_NAME = 'eu-west-2'
ECS_CLUSTER = 'DevCluster'
CONTAINER_NAME = 'test'  # This should match the name in the task definition
ECR_REPOSITORY = 'nb-runner-ecr-test'
IMAGE_TAG = 'latest'  # or whichever tag you're using

# List of commands to run containers with different arguments
commands = ["Bond_index_returns.ipynb"]

# Initialize a boto3 client
ecs_client = boto3.client('ecs', region_name=REGION_NAME)
ecr_client = boto3.client('ecr', region_name=REGION_NAME)

def get_image_uri(repository_name, image_tag):
    """
    Retrieve the image URI from ECR for the specified repository and tag.
    """
    response = ecr_client.describe_images(
        repositoryName=repository_name,
        imageIds=[{'imageTag': image_tag}]
    )
    image_uri = response['imageDetails'][0]['imageDigest']
    return f"{repository_name}@{image_uri}"

def run_task(nb_url):
    """
    Run a task on ECS with the specified command.
    """
    task_def_response = ecs_client.register_task_definition(
        family=f'my-task-{uuid.uuid4()}',
        containerDefinitions=[
            {
                'name': CONTAINER_NAME,
                'image': '263467523445.dkr.ecr.eu-west-2.amazonaws.com/nb-runner-ecr-test:latest',
                'cpu': 256,  # Adjust based on your requirements
                'memory': 512,  # Adjust based on your requirements
                'essential': True,
                'environment' : [
                    {
                        'name': 'NOTEBOOK_URL',
                        'value': nb_url
                    }
                ]
            },
        ],
        requiresCompatibilities=['FARGATE'],
        networkMode='awsvpc',
        memory='512',  # Adjust based on your requirements
        cpu='256',  # Adjust based on your requirements
        executionRoleArn='arn:aws:iam::263467523445:role/ecsTaskExecutionRole',
        taskRoleArn='arn:aws:iam::263467523445:role/ecsTaskExecutionRole',
    )

    task_definition = task_def_response['taskDefinition']['taskDefinitionArn']

    response = ecs_client.run_task(
        cluster=ECS_CLUSTER,
        launchType='FARGATE',
        taskDefinition=task_definition,
        count=1,
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': ['subnet-05078060630687d00', 'subnet-0c7fb867dd577f5e9', 'subnet-0c69f7ce71f49a66e'],  # Specify your subnets
                'assignPublicIp': 'ENABLED'
            }
        },
    )
    return response

if __name__ == '__main__':
    for cmd in commands:
        response = run_task(cmd)
        print(f"Task started with notebook {cmd}: {response}")