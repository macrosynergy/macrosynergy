from concurrent.futures import ThreadPoolExecutor
import time
import boto3
import paramiko

# Get ec2 instance with name notebook-runner-*]
ec2 = boto3.resource("ec2")
instances = ec2.instances.filter(
    Filters=[
        {"Name": "tag:Name", "Values": ["notebook-runner-*"]},
    ]
)

print(f"Found {len(list(instances))} instances")


# Get the number of ipynb files in the s3 bucket notebooks
s3 = boto3.resource("s3")
bucket = s3.Bucket("macrosynergy-notebook-prod")
objects_info = [(obj.key, obj.size) for obj in bucket.objects.all()]

# Filter for notebook files
notebooks_info = [(key, size) for key, size in objects_info if key.endswith(".ipynb")]
notebooks = [obj.key for obj in bucket.objects.filter(Prefix="")]
notebooks = [notebook for notebook in notebooks if notebook.endswith(".ipynb")]
sorted_notebooks_info = sorted(notebooks_info, key=lambda x: x[1])
notebooks = [name for name, size in sorted_notebooks_info]
print(f"Found {len(notebooks)} notebooks in the s3 bucket")

batch_size = len(notebooks) // len(list(instances))
# Get remainder too
remainder = len(notebooks) % len(list(instances))

# Batch the notebooks
batches = []
for i in range(len(list(instances))):
    batches.append(notebooks[i::len(list(instances))])
    # if remainder > 0:
    #     batches.append(notebooks[i * batch_size : (i + 1) * batch_size + 1])
    #     remainder -= 1
    # else:
    #     batches.append(notebooks[i * batch_size : (i + 1) * batch_size])
bucket_url = "https://macrosynergy-notebook-prod.s3.eu-west-2.amazonaws.com/"

print(len(batches))
print(len(list(instances)))

# If len(notebooks) < len(instances), then don't use all instances
# Start the ec2 instances
for instance in instances:
    instance.start()

for instance in instances:
    instance.wait_until_running()

def run_commands_on_ec2(instance, notebooks):
    try:
        instance_ip = instance.public_ip_address
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=instance_ip, username="ubuntu", key_filename="./Notebook-Runner.pem"
        )
        print("Connection Succeeded!!!")
        commands = [
            "rm -rf notebooks",
            " && ".join(
                ["wget -P notebooks/ " + bucket_url + notebook for notebook in notebooks]
            ),
            "pip install macrosynergy --upgrade",
            "python3 run_notebooks.py",
            "rm -rf notebooks",
        ]
        for command in commands:
            print("Executing {}".format(command))
            stdin, stdout, stderr = ssh_client.exec_command(command)
            print(stdout.read().decode("utf-8"))
            print(stderr.read().decode("utf-8"))
        ssh_client.close()
        instance.stop()
        return "Success"
    except:
        print("Failed to connect, retrying...")
        time.sleep(2)

def process_instance(instance):
    instance_ip = instance.public_ip_address
    print(f"Running notebooks on {instance_ip}")
    return run_commands_on_ec2(instance, batches.pop())

max_threads = min(len(list(instances)), len(batches))
with ThreadPoolExecutor(max_threads) as executor:
    # Use executor.map to run the process_instance function concurrently on each instance
    output = executor.map(process_instance, list(instances))

print(output)


# For each batch, run the notebooks on the instances in parallel

# Use the run_notebook.py script