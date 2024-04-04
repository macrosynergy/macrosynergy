from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
import boto3
import pandas as pd
import paramiko
from botocore.exceptions import ClientError
from datetime import datetime
import logging

#logging.basicConfig(level=logging.DEBUG)
#paramiko.util.log_to_file('paramiko.log')

# Get ec2 instance with name notebook-runner-* and state isnt terminated

branch_name = "test"

start_time = time.time()

aws_region = "eu-west-2"

ec2 = boto3.resource("ec2", region_name=aws_region)
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
notebooks_info = [(key, size) for key, size in objects_info if key.endswith(".ipynb") and key not in ["Signal_optimization_basics.ipynb", "Regression-based_macro_trading_signals.ipynb", "Regression-based_FX_signals.ipynb"]]
notebooks = [obj.key for obj in bucket.objects.filter(Prefix="")]
notebooks = [notebook for notebook in notebooks if notebook.endswith(".ipynb")]
sorted_notebooks_info = sorted(notebooks_info, key=lambda x: x[1])
notebooks = [name for name, size in sorted_notebooks_info]
print(f"Found {len(notebooks)} notebooks in the s3 bucket")

# Uncomment if you want to run a small batch for test purposes
#batch_size = 1

# Batch the notebooks
batches = []
for i in range(len(list(instances))):
    batch = notebooks[i::len(list(instances))]
#    batch = batch[:batch_size]
    batches.append(batch)
bucket_url = os.getenv("AWS_NOTEBOOK_BUCKET")

# If len(notebooks) < len(instances), then don't use all instances
# Start the ec2 instances

for instance in instances:
    if instance.state["Name"] == "stopping":
        instance.wait_until_stopped()
    instance.start()

for instance in instances:
    instance.wait_until_running()

"""
1 - Attempt to connect to the instance via SSH
2 - If failed, retry in 2 seconds and if this is the 10th retry, stop the instance and throw an error
3 - If succeeded, run commands with no hangup
4 - Once commands have been completed, get the output from file and store in a dictionary
5 - Stop the instance
6 - Send an email with the output

NOTE: If commands have no hangup then need to find a way to know when the commands have been completed
Suggestion for now is that only once the entire python script is complete does it create the output file which is then deleted once the file has been read
"""

def connect_to_instance(instance):
    retries = 0
    while retries < 10:
        try:
            instance_ip = instance.public_ip_address
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            print(f"Connecting to {instance.id}")
            ssh_client.connect(
                hostname=instance_ip, username="ubuntu", key_filename="./notebook_runner.pem"
            )
            if not ssh_client.get_transport().is_active():
                raise Exception("Failed to connect to instance")
            print(f"Opening session to {instance.id}")
            ssh_client.get_transport().open_session(timeout=10)
            print(f"Connection to {instance.id} Succeeded!!!")
            return ssh_client
        except:
            if ssh_client:
                ssh_client.close()
            retries += 1
            print("Failed to connect, retrying...")
            time.sleep(2)
    print("Failed to connect after 10 retries, stopping instance...")
    instance.stop()
    raise Exception("Failed to connect to instance")

def run_commands_on_ec2(instance, notebooks):
    ssh_client = connect_to_instance(instance)
    outputs = {"succeeded": [], "failed": []}
    try:
        # Initial cleanup commands
        cleanup_commands = "rm -rf notebooks failed_notebooks.txt successful_notebooks.txt nohup.out myvenv"
        print(f"Running cleanup commands on {instance.id}...")
        stdin, stdout, stderr = ssh_client.exec_command(cleanup_commands, timeout=3)
        stdout.channel.recv_exit_status()
        print(f"Cleanup commands completed on {instance.id}")

        print(f"Running wget commands on {instance.id}...")
        for notebook in notebooks:
            # Wget commands
            wget_command = "wget -P notebooks/ " + bucket_url + notebook 
            stdin, stdout, stderr = ssh_client.exec_command(wget_command, timeout=3)
            stdout.channel.recv_exit_status()
        print(f"Wget commands completed on {instance.id}")
        # Consider adding a delay or checking for command completion if necessary
        
        print(f"Running venv commands on {instance.id}...")
        venv_commands = "python3 -m venv myvenv"
        stdin, stdout, stderr = ssh_client.exec_command(venv_commands, timeout=50)
        stdout.channel.recv_exit_status()
        print(f"Venv commands completed on {instance.id}")

        # Pip and nohup commands
        print(f"Running pip commands on {instance.id}...")
        pip_commands = f"myvenv/bin/python -m pip install linearmodels jupyter nbformat git+https://github.com/macrosynergy/macrosynergy@{branch_name}  --upgrade"
        stdin, stdout, stderr = ssh_client.exec_command(pip_commands, timeout=50)
        stdout.channel.recv_exit_status()
        print(f"Pip commands completed on {instance.id}")

        notebook_runner_cmd = "nohup myvenv/bin/python run_notebooks.py > nohup.out 2>&1 &"
        print(f"Running notebook runner commands on {instance.id}...")
        ssh_client.exec_command(notebook_runner_cmd, timeout=50)
        
        print(f"Getting output from instance... {instance.id}")
        successful_notebooks, failed_notebooks = get_output_from_instance(ssh_client)
        outputs["succeeded"].extend(successful_notebooks)
        outputs["failed"].extend(failed_notebooks)
    finally:
        print(f"Downloading logs from instance... {instance.id}")

        timestamp = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d.%H:%M:%S')

        log_commands = f"aws s3 cp notebooks/ s3://{os.getenv("AWS_NB_RUNNER_LOGS")}/{instance.id}_{timestamp}/notebooks --recursive"
        stdin, stdout, stderr = ssh_client.exec_command(log_commands, timeout=50)
        stdout.channel.recv_exit_status()

        log_commands = f"aws s3 cp nohup.out s3://{os.getenv("AWS_NB_RUNNER_LOGS")}/{instance.id}_{timestamp}/nohup.out"
        stdin, stdout, stderr = ssh_client.exec_command(log_commands, timeout=50)
        stdout.channel.recv_exit_status()

        print(f"Stopping instance... {instance.id}")
        ssh_client.close()
        instance.stop()
    return outputs

def check_output(ssh_client):
    command = "ls"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    output = stdout.read().decode("utf-8")
    error = stderr.read().decode("utf-8")
    return "failed_notebooks.txt" in output and "successful_notebooks.txt" in output
    

def get_output_from_instance(ssh_client):
    python_running = True
    start_time = time.time()
    while python_running:
        #print(f"Python process is still running on instance, waiting for it to finish...")
        command = "ps -ef | grep python"
        stdin, stdout, stderr = ssh_client.exec_command(command)
        output = stdout.read().decode("utf-8")
        error = stderr.read().decode("utf-8")

        if "run_notebooks.py" in output:
            if time.time() - start_time > 6000:
                print("Python process has been running for over 100 minutes, stopping instance...")
                python_running = False
            time.sleep(5)
        else:
            print("Python process has finished")
            command = "cat nohup.out"
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=10)
            output = stdout.read().decode("utf-8")
            error = stderr.read().decode("utf-8")
            print(output)
            python_running = False
    
    print("Python process has finished, getting failed notebooks...")
    command = "cat failed_notebooks.txt"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    output = stdout.read().decode("utf-8")
    error = stderr.read().decode("utf-8")
    failed_notebooks = output.split("\n")[:-1] # Removes empty string at the end

    print("Python process has finished, getting successful notebooks...")
    command = "cat successful_notebooks.txt"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    output = stdout.read().decode("utf-8")
    error = stderr.read().decode("utf-8")
    successful_notebooks = output.split("\n")[:-1] # Removes empty string at the end
    
    return successful_notebooks, failed_notebooks

def process_instance(instance):
    instance_ip = instance.public_ip_address
    print(f"Running notebooks on {instance.id}")
    try:
        batch = batches.pop()
        output = run_commands_on_ec2(instance, batch)
    except Exception as e:
        print(f"Failed to run notebooks: {batch} on {instance.id}")
        print(e)
        instance.stop()
        return {"succeeded": [], "failed": batch}
    return output

max_threads = min(len(list(instances)), len(batches))
with ThreadPoolExecutor(max_threads) as executor:
    # Use executor.map to run the process_instance function concurrently on each instance
    output = executor.map(process_instance, list(instances))

print("FINSIHED RUNNING ALL NOTEBOOKS")
merged_dict = {}

# Merge dictionaries
for d in list(output):
    for key, value in d.items():
        if key in merged_dict:
            merged_dict[key].extend(value)
        else:
            merged_dict[key] = value

print(f"Successful notebooks: {merged_dict['succeeded']}\nFailed notebooks: {merged_dict['failed']}")

end_time = time.time()

def send_email(subject, body, recipient, sender):
    # Specify your AWS region
    aws_region = "eu-west-2"

    # Create an SES client
    ses_client = boto3.client("ses", region_name=aws_region)

    # Specify the email content
    email_content = {"Subject": {"Data": subject}, "Body": {"Html": {"Data": body}}}

    try:
        # Send the email
        response = ses_client.send_email(
            Source=sender,
            Destination={"ToAddresses": recipient},
            Message=email_content,
        )
        print(f"Email sent! Message ID: {response['MessageId']}")

    except ClientError as e:
        print(f"Error sending email: {e.response['Error']['Message']}")

if len(merged_dict["failed"]) == 0:
    exit(0)
else:
    email_subject = "Notebook Failures"
    email_body = f"Please note that the following notebooks failed when ran on the branch {branch_name}: \n{pd.DataFrame(merged_dict['failed']).to_html()}\nThe total time to run all notebooks was {end_time - start_time} seconds."
    recipient_email = os.getenv("EMAIL_RECIPIENTS").split(",")
    sender_email = os.getenv("SENDER_EMAIL")

    send_email(email_subject, email_body, recipient_email, sender_email)
    exit(1)