import asyncio
import os
import requests
from pydantic.main import BaseModel
import re
import tensorflow as tf
import logging

# Log format
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) 

# server_status Address
inform_SE: str = 'http://ccljhub.gachon.ac.kr:40019/FLSe/'

# FL Client Status class
class FL_client_status(BaseModel):
    FL_client_num: int =  0 # FL client ID
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None # FL server IP
    FL_round: int = 1 # round
    FL_loss: int = 0 # 
    FL_accuracy: int = 0 
    FL_next_gl_model: int = 0 


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

    s = listdir[0] 
    p = re.compile(r'\d+')  # Select Number Pattern
    local_list_sorted = sorted(listdir, key=lambda s: int(p.search(s).group()))  # sorting gl_model_version

    local_model_name = local_list_sorted[len(local_list_sorted) - 1]  # select latest gl_model
    model = tf.keras.models.load_model(f'/local_model/{local_model_name}')
    
    logging.info(f'local_model_name: {local_model_name}')

    return model

# check train finish info to client manager
async def notify_fin():

    FL_client_start = False

    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFin')
    r = await future2
    logging.info('try notify_fin')
    if r.status_code == 200:
        logging.info('trainFin')
    else:
        logging.error(f'notify_fin error: {r.content}')

    return FL_client_start


# check train fail info to client manager
async def notify_fail():

    logging.info('notify_fail start')

    FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFail')
    r = await future1
    if r.status_code == 200:
        logging.error('trainFin')
    else:
        logging.error('notify_fail error: ', r.content)
    
    return FL_client_start