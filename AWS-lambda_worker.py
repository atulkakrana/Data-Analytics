import boto3
import paramiko
def lambda_handler(event, context):

    s3_client = boto3.client('s3')
    #Download private key file from secure S3 bucket
    s3_client.download_file('atulstore','Linux-test.pem', '/tmp/Linux-test.pem')

    k = paramiko.RSAKey.from_private_key_file("/tmp/Linux-test.pem")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    host=event['IP']
    print ("Connecting to " + host)
    c.connect( hostname = host, username = "ubuntu", pkey = k )
    print ("Connected to " + host)

    commands = [
        "aws s3 cp s3://atulstore/analysis.sh /home/ubuntu/analysis.sh",
        "chmod 700 /home/ubuntu/analysis.sh",
        "/home/ubuntu/analysis.sh"
        ]
    for command in commands:
        print("Executing {}".format(command))
        stdin,stdout, stderr = c.exec_command(command)
        print(stdout.read())
        print(stderr.read())

    return
    {
        'message' : "Script execution completed. See Cloudwatch logs for complete output"
    }
