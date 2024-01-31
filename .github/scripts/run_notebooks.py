from concurrent.futures import ThreadPoolExecutor
import re
import time
import boto3
import paramiko

# Get ec2 instance with name notebook-runner-* and state isnt terminated

ec2 = boto3.resource("ec2")
instances = ec2.instances.filter(
    Filters=[
        {"Name": "tag:Name", "Values": ["notebook-runner-*"]},
        {"Name": "instance-state-name", "Values": ["running", "stopped", "pending", "stopping", "starting"]}
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
batch_size = 3
# Get remainder too
remainder = len(notebooks) % len(list(instances))

# Batch the notebooks
batches = []
for i in range(len(list(instances))):
    batch = notebooks[i::len(list(instances))]
    batch = batch[:batch_size]
    batches.append(batch)
bucket_url = "https://macrosynergy-notebook-prod.s3.eu-west-2.amazonaws.com/"

print(len(batches))
print(len(batches[0]))
print(len(list(instances)))

# If len(notebooks) < len(instances), then don't use all instances
# Start the ec2 instances
for instance in instances:
    instance.start()

for instance in instances:
    instance.wait_until_running()

def run_commands_on_ec2(instance, notebooks):
    connected = False
    while not connected:
        try:
            instance_ip = instance.public_ip_address
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                hostname=instance_ip, username="ubuntu", key_filename="./Notebook-Runner.pem"
            )
            print("Connection Succeeded!!!")
            connected = True
            outputs = {"succeeded": [], "failed": []}
            commands = [
                "rm -rf notebooks",
                " && ".join(
                    ["wget -P notebooks/ " + bucket_url + notebook for notebook in notebooks]
                ),
                "source myvenv/bin/activate \n pip install linearmodels --upgrade \n pip install jupyter --upgrade \n pip install macrosynergy --upgrade \n python3 run_notebooks.py",
                "rm -rf notebooks"
            ]
            for command in commands:
                print("Executing {}".format(command))
                stdin, stdout, stderr = ssh_client.exec_command(command)
                output = stdout.read().decode("utf-8")
                error = stderr.read().decode("utf-8")
                print(output)
                print(error)
                if command == "source myvenv/bin/activate \n pip install linearmodels --upgrade \n pip install jupyter --upgrade \n pip install macrosynergy --upgrade \n python3 run_notebooks.py":
                    # Find each occurence of "Notebook x succeeded" and "Notebook x failed" and store in outputs
                    success_regex = r"Notebook (.+) succeeded"
                    failure_regex = r"Notebook (.+) failed"

                    for match in re.finditer(success_regex, error):
                        notebook = match.group(1)
                        print(notebook)
                        notebook = notebook.split("/")[-1]
                        outputs["succeeded"].append(notebook)
                    for match in re.finditer(failure_regex, error):
                        notebook = match.group(1)
                        print(notebook)
                        notebook = notebook.split("/")[-1]
                        outputs["failed"].append(notebook)

            ssh_client.close()
            instance.stop()
            return outputs
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

merged_dict = {}

# Merge dictionaries
for d in list(output):
    for key, value in d.items():
        if key in merged_dict:
            merged_dict[key].extend(value)
        else:
            merged_dict[key] = value

print(merged_dict)

# For each batch, run the notebooks on the instances in parallel

# Use the run_notebook.py script