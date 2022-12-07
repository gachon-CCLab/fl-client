import asyncio
import os
import requests
from pydantic.main import BaseModel
import re
import tensorflow as tf
import logging

# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) 

# server_status 주소
inform_SE: str = 'http://ccljhub.gachon.ac.kr:40019/FLSe/'

# FL Client 상태 class
class FL_client_status(BaseModel):
    FL_client_num: int =  0 # FL client 번호(ID)
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None # FL server IP
    FL_round: int = 1 # 현재 수행 round
    FL_loss: int = 0 # 성능 loss
    FL_accuracy: int = 0 # 성능 acc
    FL_next_gl_model: int = 0 # 글로벌 모델 버전


# send client name to server_status
def register_client():
    client_name = os.uname()[1]

    res = requests.put(inform_SE + 'RegisterClient', params={'ClientName': client_name})
    if res.status_code == 200:
        client_num = res.json()['client_num']

    return client_num


# make local model directory
def make_model_directory():

    # Local Model repository
    if not os.path.isdir('./local_model'):
        os.mkdir('./local_model')
    else:
        pass

# latest local model download
def download_local_model(listdir):
    # mac에서만 시행 (.DS_Store 파일 삭제)
    if '.DS_Store' in listdir:
        i = listdir.index(('.DS_Store'))
        del listdir[i]

    s = listdir[0]  # 비교 대상(gl_model 지정) => sort를 위함
    p = re.compile(r'\d+')  # 숫자 패턴 추출
    local_list_sorted = sorted(listdir, key=lambda s: int(p.search(s).group()))  # gl model 버전에 따라 정렬

    local_model_name = local_list_sorted[len(local_list_sorted) - 1]  # 최근 gl model 추출
    model = tf.keras.models.load_model(f'/model/{local_model_name}')
    
    # local_model_v = int(local_model_name.split('_')[1])
    logging.info(f'local_model_name: {local_model_name}')

    return model

# check train finish info to client manager
async def notify_fin():
    global status

    status.FL_client_start = False

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFin')
    r = await future2
    logging.info('try notify_fin')
    if r.status_code == 200:
        logging.info('trainFin')
    else:
        logging.error(f'notify_fin error: {r.content}')
    return status


# check train fail info to client manager
async def notify_fail():
    global status

    logging.info('notify_fail start')

    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFail')
    r = await future1
    if r.status_code == 200:
        logging.error('trainFin')
    else:
        logging.error('notify_fail error: ', r.content)
    
    return status