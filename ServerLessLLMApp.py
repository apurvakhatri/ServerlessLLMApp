import boto3
import json
import uuid # Used to make transcription job name unique
import time
import securing_credentials
#Imports for audio files
from pydub import AudioSegment
from pydub.playback import play


# username
ACCESS_KEY = securing_credentials.ACCESS_KEY
# password
SECRET_KEY = securing_credentials.SECRET_KEY

#bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-2')
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1',aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)
prompt = "Write a summary of India."

kwargs = {
    "modelId": "amazon.titan-text-lite-v1", #LLM from Amazon
    "contentType": "application/json", #Mine type of data we are going to send in i.e input in our request
    "accept": "*/*", # Mine type of data we're going to receive well i.e the output [json by default]
    "body": json.dumps(
        {
            "inputText": prompt
        }
    )
}

response = bedrock_runtime.invoke_model(**kwargs) #Body of the response returned gives a pointer to a streaming boto3 object where the actual data is stored.
response_body = json.loads(response.get('body').read())
generated_output = response_body['results'][0]['outputText']
print(generated_output)

# Load an MP3 file
audio = AudioSegment.from_mp3("inquiry.mp3")

# Play the audio
#play(audio)


# Creating transcription using audio files


# Creating S3 client

s3_client = boto3.client('s3', region_name = 'us-west-2', aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)

def upload_data_to_s3(boto_s3_client, bucket_name, upload_file, upload_file_name):
    boto_s3_client.upload_file(upload_file, bucket_name, upload_file_name)


bucket_name = 'audiorecordingsllm' # S3 bucket name
upload_object = 'inquiry.mp3'

# upload_data_to_s3(s3_client, bucket_name, upload_object, upload_object)

transcribe_client = boto3.client('transcribe', region_name = 'us-east-1', aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY) # Transcribe client

job_name = 'transcription-job-' + str(uuid.uuid4()) # transcription job needs to be unique

def transcribe_audio(transcribe_client, job_name, bucket_name, upload_object):
    transcribe_response = transcribe_client.start_transcription_job(
        TranscriptionJobName = job_name,
        Media = {
            'MediaFileUri': f's3://{bucket_name}/{upload_object}'
        },
        MediaFormat = 'mp3',
        LanguageCode = 'en-US',
        OutputBucketName = bucket_name,
        Settings = {
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 2
        }
    )
    return transcribe_response

#transcribe_response = transcribe_audio(transcribe_client, job_name, bucket_name, upload_object)

job_name = 'transcription-job-842c897f-ee70-4f1b-831a-08e9a80e9df8' #Overwrite

status = transcribe_client.get_transcription_job(TranscriptionJobName = job_name)
while True:
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    time.sleep(2)

print(status['TranscriptionJob']['TranscriptionJobStatus'])

output_text = ""

if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
    # Load the transcript from S3
    transcript_key = f"{job_name}.json"
    transcript_obj = s3_client.get_object(Bucket=bucket_name, Key = transcript_key)
    transcript_text = transcript_obj['Body'].read().decode('utf-8')
    transcript_json = json.loads(transcript_text)
    pretty_json = json.dumps(transcript_json, indent=4)
    # print(transcript_json)

    current_speaker = None

    items = transcript_json['results']['items']

    for item in items:
        speaker_label = item.get('speaker_label', None)
        content = item['alternatives'][0]['content']

        if speaker_label is not None and speaker_label != current_speaker:
            current_speaker = speaker_label
            output_text += f"\n{current_speaker}: "

        if item['type'] == 'punctuation':
            output_text = output_text.rstrip()

        output_text += f"{content} "

    print(output_text)