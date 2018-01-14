from __future__ import print_function
import boto3

def lambda_handler(event, context):
    #Get IP addresses of EC2 instances
    client = boto3.client('ec2')
    print(client)
    instDict=client.describe_instances(Filters=[{'Name':'tag:Environment','Values':['Dev']}])

    hostList=[]
    for r in instDict['Reservations']:
        for inst in r['Instances']:
            hostList.append(inst['PublicIpAddress'])
    
    print(hostList)

    #Invoke worker function for each IP address
    client = boto3.client('lambda')
    for host in hostList:
        print ("Invoking worker_function on " + host)
        invokeResponse=client.invoke(
            FunctionName='worker_function',
            InvocationType='Event',
            LogType='Tail',
            Payload='{"IP":"'+ host +'"}'
        )
        print("Message from Atul")
        print("Invoked",invokeResponse)

    return{
        'message' : "Trigger function finished"
    }