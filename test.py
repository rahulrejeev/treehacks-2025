import boto3
from dotenv import load_dotenv
import os
import json
from time import time

load_dotenv()
try:
    AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION=os.getenv('AWS_REGION')
    AWS_BUCKET_NAME=os.getenv('AWS_BUCKET_NAME')
    s3 = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY, 
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                      region_name=AWS_REGION)
except Exception as e:
    print(f"Error: {e}")
    raise e

# Define the five JSON telemetry payloads as Python dictionaries
payloads = [
    # 1. Calm
    {
        "user": {
            "user_id": "75fe4fd8-28fa-49bb-a26e-4ad4c9d92ee5",
            "provider": "FITBIT",
            "active": True,
            "scopes": None,
            "reference_id": "",
            "created_at": None,
            "last_webhook_update": None
        },
        "data": [
            {
                "metadata": {
                    "start_time": "2025-02-16T14:00:00.000000+00:00",
                    "end_time": "2025-02-16T14:10:00.000000+00:00",
                    "emotional_state": "calm"
                },
                "heart_data": {
                    "heart_rate_data": {
                        "summary": {
                            "avg_hr_bpm": 65,
                            "resting_hr_bpm": 60,
                            "max_hr_bpm": 70,
                            "min_hr_bpm": 60,
                            "avg_hrv_rmssd": 90
                        },
                        "detailed": None
                    }
                },
                "oxygen_data": {
                    "avg_saturation_percentage": 98.0,
                    "vo2max_ml_per_min_per_kg": None,
                    "vo2_samples": None,
                    "saturation_samples": None
                },
                "blood_pressure_data": None,
                "hydration_data": None,
                "temperature_data": None,
                "glucose_data": None,
                "ketone_data": None,
                "device_data": None,
                "measurements_data": None
            }
        ],
        "type": "body"
    },
    # 2. Restless
    {
        "user": {
            "user_id": "75fe4fd8-28fa-49bb-a26e-4ad4c9d92ee5",
            "provider": "FITBIT",
            "active": True,
            "scopes": None,
            "reference_id": "",
            "created_at": None,
            "last_webhook_update": None
        },
        "data": [
            {
                "metadata": {
                    "start_time": "2025-02-16T14:15:00.000000+00:00",
                    "end_time": "2025-02-16T14:25:00.000000+00:00",
                    "emotional_state": "restless"
                },
                "heart_data": {
                    "heart_rate_data": {
                        "summary": {
                            "avg_hr_bpm": 80,
                            "resting_hr_bpm": 75,
                            "max_hr_bpm": 90,
                            "min_hr_bpm": 75,
                            "avg_hrv_rmssd": 60
                        },
                        "detailed": None
                    }
                },
                "oxygen_data": {
                    "avg_saturation_percentage": 97.0,
                    "vo2max_ml_per_min_per_kg": None,
                    "vo2_samples": None,
                    "saturation_samples": None
                },
                "blood_pressure_data": None,
                "hydration_data": None,
                "temperature_data": None,
                "glucose_data": None,
                "ketone_data": None,
                "device_data": None,
                "measurements_data": None
            }
        ],
        "type": "body"
    },
    # 3. Angry
    {
        "user": {
            "user_id": "75fe4fd8-28fa-49bb-a26e-4ad4c9d92ee5",
            "provider": "FITBIT",
            "active": True,
            "scopes": None,
            "reference_id": "",
            "created_at": None,
            "last_webhook_update": None
        },
        "data": [
            {
                "metadata": {
                    "start_time": "2025-02-16T14:30:00.000000+00:00",
                    "end_time": "2025-02-16T14:40:00.000000+00:00",
                    "emotional_state": "angry"
                },
                "heart_data": {
                    "heart_rate_data": {
                        "summary": {
                            "avg_hr_bpm": 105,
                            "resting_hr_bpm": 100,
                            "max_hr_bpm": 110,
                            "min_hr_bpm": 95,
                            "avg_hrv_rmssd": 40
                        },
                        "detailed": None
                    }
                },
                "oxygen_data": {
                    "avg_saturation_percentage": 95.0,
                    "vo2max_ml_per_min_per_kg": None,
                    "vo2_samples": None,
                    "saturation_samples": None
                },
                "blood_pressure_data": None,
                "hydration_data": None,
                "temperature_data": None,
                "glucose_data": None,
                "ketone_data": None,
                "device_data": None,
                "measurements_data": None
            }
        ],
        "type": "body"
    },
    # 4. Somewhat Relaxed
    {
        "user": {
            "user_id": "75fe4fd8-28fa-49bb-a26e-4ad4c9d92ee5",
            "provider": "FITBIT",
            "active": True,
            "scopes": None,
            "reference_id": "",
            "created_at": None,
            "last_webhook_update": None
        },
        "data": [
            {
                "metadata": {
                    "start_time": "2025-02-16T14:45:00.000000+00:00",
                    "end_time": "2025-02-16T14:55:00.000000+00:00",
                    "emotional_state": "somewhat relaxed"
                },
                "heart_data": {
                    "heart_rate_data": {
                        "summary": {
                            "avg_hr_bpm": 75,
                            "resting_hr_bpm": 70,
                            "max_hr_bpm": 80,
                            "min_hr_bpm": 70,
                            "avg_hrv_rmssd": 70
                        },
                        "detailed": None
                    }
                },
                "oxygen_data": {
                    "avg_saturation_percentage": 97.5,
                    "vo2max_ml_per_min_per_kg": None,
                    "vo2_samples": None,
                    "saturation_samples": None
                },
                "blood_pressure_data": None,
                "hydration_data": None,
                "temperature_data": None,
                "glucose_data": None,
                "ketone_data": None,
                "device_data": None,
                "measurements_data": None
            }
        ],
        "type": "body"
    },
    # 5. Calm (Final)
    {
        "user": {
            "user_id": "75fe4fd8-28fa-49bb-a26e-4ad4c9d92ee5",
            "provider": "FITBIT",
            "active": True,
            "scopes": None,
            "reference_id": "",
            "created_at": None,
            "last_webhook_update": None
        },
        "data": [
            {
                "metadata": {
                    "start_time": "2025-02-16T15:00:00.000000+00:00",
                    "end_time": "2025-02-16T15:10:00.000000+00:00",
                    "emotional_state": "calm"
                },
                "heart_data": {
                    "heart_rate_data": {
                        "summary": {
                            "avg_hr_bpm": 65,
                            "resting_hr_bpm": 60,
                            "max_hr_bpm": 70,
                            "min_hr_bpm": 60,
                            "avg_hrv_rmssd": 90
                        },
                        "detailed": None
                    }
                },
                "oxygen_data": {
                    "avg_saturation_percentage": 98.5,
                    "vo2max_ml_per_min_per_kg": None,
                    "vo2_samples": None,
                    "saturation_samples": None
                },
                "blood_pressure_data": None,
                "hydration_data": None,
                "temperature_data": None,
                "glucose_data": None,
                "ketone_data": None,
                "device_data": None,
                "measurements_data": None
            }
        ],
        "type": "body"
    }
]
def run_mock():
    # Loop through each payload and upload it to S3
    for index, payload in enumerate(payloads, start=1):
        # Use the emotional state to create a meaningful filename
        emotional_state = payload["data"][0]["metadata"]["emotional_state"].replace(" ", "_")
        filename = f"telemetry_{index}_{emotional_state}.json"
        
        # Convert the dictionary to a JSON string with indentation for readability
        json_data = json.dumps(payload, indent=4)
        
        # Upload the JSON string as an object to S3
        response = s3.put_object(
            Bucket=AWS_BUCKET_NAME,
            Key=filename,
            Body=json_data,
            ContentType='application/json'
        )
        status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        print(f"Uploaded {filename} to bucket '{AWS_BUCKET_NAME}' with status code: {status_code}")
        time.sleep(40)
        print("Waiting for 60 seconds...")

if __name__ == "__main__":
    run_mock()