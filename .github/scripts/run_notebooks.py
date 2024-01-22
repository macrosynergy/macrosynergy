# RAN AT 13:24
import time
import boto3
import paramiko

# Get ec2 instance with name notebook-runner-*]
ec2 = boto3.resource("ec2")
instances = ec2.instances.filter(
    Filters=[
        {"Name": "tag:Name", "Values": ["notebook-runner-*"]},
        {"Name": "instance-state-name", "Values": ["running", "stopped"]},
    ]
)

print(f"Found {len(list(instances))} instances")


# Get the number of ipynb files in the s3 bucket notebooks
s3 = boto3.resource("s3")
bucket = s3.Bucket("macrosynergy-notebook-prod")
notebooks = [obj.key for obj in bucket.objects.filter(Prefix="")]
notebooks = [notebook for notebook in notebooks if notebook.endswith(".ipynb")]
print(f"Found {len(notebooks)} notebooks in the s3 bucket")

batch_size = len(notebooks) // len(list(instances))
batch_size = 1
# Get remainder too
remainder = len(notebooks) % len(list(instances))
remainder = 0

# Batch the notebooks
batches = []
for i in range(len(list(instances))):
    if remainder > 0:
        batches.append(notebooks[i * batch_size : (i + 1) * batch_size + 1])
        remainder -= 1
    else:
        batches.append(notebooks[i * batch_size : (i + 1) * batch_size])
bucket_url = "https://macrosynergy-notebook-prod.s3.eu-west-2.amazonaws.com/"

# If len(notebooks) < len(instances), then don't use all instances
# Start the ec2 instances
# for instance in instances:
#     instance.start()

list(instances)[0].start()

# Add a sleep here to wait for the instances to start


def run_commands_on_ec2(instance_ip, notebooks):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(
        hostname=instance_ip, username="ubuntu", key_filename="./Notebook-Runner.pem"
    )
    commands = [
        "rm -rf notebooks",
        "mkdir notebooks",
        " && ".join(
            ["wget -P notebooks/ " + bucket_url + notebook for notebook in notebooks]
        ),
        "pip install macrosynergy --upgrade",
        "python3 run_notebooks.py",
        # "rm -rf notebooks",
    ]
    for command in commands:
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(stdout.read())
        print(stderr.read())
    ssh_client.close()


# Run the notebooks on the instances
# for instance in instances:
#     # Wait until the instance state is running
#     while instance.state['Name'] != 'running':
#         time.sleep(2)
#     run_commands_on_ec2(instance.public_ip_address, batches.pop())

while list(instances)[0].state["Name"] != "running":
    time.sleep(2)
time.sleep(2)
run_commands_on_ec2(list(instances)[0].public_ip_address, batches.pop())

# for instance in instances:
#     instance.stop()

list(instances)[0].stop()


# For each batch, run the notebooks on the instances in parallel

# Use the run_notebook.py script
