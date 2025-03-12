import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 0. load key file------------------
def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()

def set_api_key(api_key_file):
    '''
    openai 환경변수 설정 함
    api_key_file : api_key.txt의 경로
    '''

    # API 키 로드 및 환경변수 설정
    openai.api_key = load_file(api_key_file)
    os.environ['OPENAI_API_KEY'] = openai.api_key


# 1-1 audio2text--------------------

def transcribe_audio(filename):
    '''
    OpenAI를 통한 파일 변환
    filename : audio 파일 경로 (단일)
    '''

    # OpenAI 클라이언트 생성
    client = OpenAI()

    # 오디오 파일을 읽어서, 위스퍼를 사용한 변환
    audio_file = open(filename, "rb")

    # 결과 반환
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language="ko",
        response_format="text",
    )

    return transcript

def audio2text(filenames, result=True):
    '''
    filenames : audio 파일 경로 혹은 경로 리스트
    '''
    # filenames가 리스트가 아닐 경우
    if not isinstance(filenames, list):
        filenames = [filenames]
    
    # 데이터프레임 정의
    df = pd.DataFrame(columns=['filename', 'text'])

    for filename in filenames:
        text = transcribe_audio(filename)
        df = pd.concat([df, pd.DataFrame({'filename': [filename], 'text': [text]})], ignore_index=True)

    return df

# 1-2 text2summary------------------

def text2summary(input_text):

    # OpenAI 클라이언트 생성
    client = OpenAI()

    # 시스템 역할과 응답 형식 지정
    system_role = '''너는 응급상황에서 상대방의 말을 요약하는 어시스턴트입니다.
    주어진 문장에서 핵심 상황만 간결하게 요약하고, 대처 방법이나 추가 설명은 생략해 주세요.
    추가로 텍스트 요약과 함께 중요 키워드를 5개 도출해주세요.
    응답은 다음의 형식을 지켜주세요.
    {"summary": \"텍스트 요약\", "keyword": \"텍스트 중요 키워드\"}
    '''

    # 입력데이터를 GPT-3.5-turbo에 전달하고 답변 받아오기
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": input_text
            }
        ]
    )

    # 응답 받기
    answer = response.choices[0].message.content

    # 응답형식을 정리하고 return
    answer = json.loads(answer)
    answer = answer['summary']+' '+answer['keyword']

    return answer


# 2. model prediction------------------

# 데이터 예측 함수
def predict(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} # 각 텐서를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()

    return pred, probabilities

def severity_prediction(text, save_directory, verbose=False):
    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    predicted_class, probabilities = predict(text, model, tokenizer)
    predicted_class += 1

    if verbose:
        print(f"Predicted Class: {predicted_class}등급")
        print(f"Probabilities: {probabilities.tolist()[0]}")
        print(f"Patient Condition: {text}")

    return predicted_class


# 3-1. get_distance------------------
def get_dist(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key, verbose=False):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",    # 목적지 (경도, 위도)
        "option": "trafast"  # 실시간 빠른 길 옵션
    }
    
    # 밀리초(ms)를 시(h)분(m)로
    def sec_to_hms(duration):
        duration = duration//1000
        minute = (duration//60)%60
        hour = duration//60//60

        hms_str = ''
        
        if hour > 0: hms_str = str(hour) + '시간 ' + str(minute+1) + '분 '
        elif minute > 0: hms_str = str(minute+1) + '분'
        else: hms_str = '1분'

        return hms_str
    # 거리
    dist = None

    # 요청하고, 답변 받아오기
    response = requests.get(url, headers=headers, params = params)
    res = response.status_code # HTTP 상태 코드

    # 길찾기 데이터
    root = json.loads(response.text)
    
    if res != 200: # url이 응답하지 않을 경우
        if verbose:
            print(response, ": 응답하지 않습니다")

    elif root['code'] == 0 : # 길찾기가 성공하면
        distance = root['route']['trafast'][0]['summary']['distance']  # m(미터)
        duration = root['route']['trafast'][0]['summary']['duration']  # 밀리초(ms)
        hms_str= sec_to_hms(duration)
        dist = round(distance/1000, 2)
        if verbose:
            print(hms_str)

    else: # 길찾기가 성공하지 않는 경우 설명 메시지 출력
        if verbose:
            print(response['message'])

    return dist, hms_str


# 3-2. recommendation------------------
def recommend_hospital3(start_lat, start_lng, map_key_file, data, alpha=[0.1, 0.1]):
    '''
        자동차 도로로 가장 가까운 3개의 병원 출력

        start_lat: 출발지 경도
        start_lng: 출발지 위도
        map_key_file: 네이버 지도 API '아이디, 키' 파일 경로
        data: 응급실 정보 데이터 프레임(위도와 경도 필요)
        alpha: 기본값 lat, lng = [0.1, 0.1]
    '''

    # map id, map key 가져오기
    def load_file(filepath):
        with open(filepath, 'r') as file:
            return file.readline().strip()

    c_id, c_key = load_file(map_key_file).split(',')
    c_id, c_key = c_id.strip(), c_key.strip()

    # alpha값 설정
    alpha_lat, alpha_lng = alpha

    # 범위구간 정하고 해당 구간(lat, long) 응급실 정보 조회
    lat = [start_lat-alpha_lat, start_lat+alpha_lat] # 위도 구간
    long = [start_lng-alpha_lng, start_lng+alpha_lng] # 경도구간
    tmp = data[(data['위도'] >= lat[0]) & (data['위도'] <= lat[1]) & (data['경도'] >= long[0]) & (data['경도'] <= long[1])].copy()

    # 거리 리스트
    dist_list = []
    time_list = []

    # 응급실별 거리 계산
    for i in range(len(tmp)):
        dist, hms_str = get_dist(start_lat, start_lng, tmp.iloc[i]['위도'], tmp.iloc[i]['경도'], c_id, c_key)
        dist_list.append(dist)
        time_list.append(hms_str)

    # 데이터프레임에 추가
    tmp['dist'] = dist_list
    tmp['time'] = time_list

    return tmp.sort_values('dist')[:3].reset_index(drop=True)