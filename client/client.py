# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

import itertools
import os, logging, json
import re
import time
from collections import Counter

import tensorflow as tf

import flwr as fl
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
# keras에서 내장 함수 지원(to_categofical())
from keras.utils.np_utils import to_categorical

from functools import partial
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel

from . import utils

# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) 

# Make TensorFlow logs less verbose
# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# CPU만 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# FL client 상태 확인
app = FastAPI()

# client_num
client_num = utils.register_client()

# FL Client Status Object
status = utils.FL_client_status
status.FL_client_num = client_num


# Define Flower client
class CifarClient(fl.client.NumPyClient):

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        global status

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # start round time
        round_start_time = time.time()

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # end round time
        round_end_time = time.time() - round_start_time  # 연합학습 종료 시간

        # Training: model excution time by round
        train_time_result = {"client_num": status.FL_client_num, "round": status.FL_round, "next_gl_model": status.FL_next_gl_model, "execution_time": round_end_time}
        json_time_result = json.dumps(train_time_result)
        logging.info(f'train_time - {json_time_result}')

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)

        status.FL_loss = history.history["loss"][len(history.history["loss"])-1]
        status.FL_accuracy = history.history["accuracy"][len(history.history["accuracy"])-1]
        results = {
            "loss": status.FL_loss,
            "accuracy": status.FL_accuracy,
            "val_loss": history.history["val_loss"][len(history.history["val_loss"])-1],
            "val_accuracy": history.history["val_accuracy"][len(history.history["val_accuracy"])-1],
        }

        # Training: model performance by round
        train_result = {"client_num": status.FL_client_num, "round": status.FL_round, "fit_loss": status.FL_loss, "fit_accuracy": status.FL_accuracy,
                        "next_gl_model": status.FL_next_gl_model}
        json_result = json.dumps(train_result)
        logging.info(f'train_performance - {json_result}')

        # make local model directory
        utils.make_model_directory()

        # save local model
        self.model.save(f'./local_model/local_model_V{status.FL_next_gl_model}.h5')

        return parameters_prime, num_examples_train, results


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        test_loss, test_accuracy = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=1024, steps=steps)
        num_examples_test = len(self.x_test)

        # Test: model performance by round
        test_result = {"client_num": status.FL_client_num, "round": status.FL_round, "test_loss": test_loss, "test_accuracy": test_accuracy, "next_gl_model": status.FL_next_gl_model}
        json_result = json.dumps(test_result)
        logging.info(f'test - {json_result}')

        # increase next round
        status.FL_round += 1

        return test_loss, num_examples_test, {"accuracy": test_accuracy}


# Create Client Local Model & Metric
def build_model():

    # define model & metrics
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]

    # model 생성
    model = Sequential()

    # Convolutional Block (Conv-Conv-Pool-Dropout)
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classifying
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=METRICS)

    return model


@app.get('/online')
def get_info():
    return status

# asynchronously start client
@app.get("/start/{Server_IP}")
async def flclientstart(background_tasks: BackgroundTasks, Server_IP: str):
    global status
    
    # client_manager 주소
    client_res = requests.get('http://localhost:8003/info/')

    # 최신 global model 버전
    latest_gl_model_v = client_res.json()['GL_Model_V']

    # 다음 global model 버전
    status.FL_next_gl_model = latest_gl_model_v + 1

    logging.info('bulid model')

    logging.info('FL start')
    status.FL_client_start = True
    status.FL_server_IP = Server_IP

    # start FL Client
    background_tasks.add_task(flower_client_start)

    return status


async def flower_client_start():
    logging.info('FL learning ready')
    global status

    # split partition => apply each client dataset
    (x_train, y_train), (x_test, y_test) = load_partition()
    logging.info('data loaded')

    # check local_model 
    local_list = os.listdir(f'/local_model')
    if not local_list:
        logging.info('init local model')
        model = build_model()

    else:
        # download latest local model
        logging.info('Latest Local Model download')
        model = utils.download_local_model(local_list)

    try:
        loop = asyncio.get_event_loop()
        client = CifarClient(model, x_train, y_train, x_test, y_test)
        request = partial(fl.client.start_numpy_client, server_address=status.FL_server_IP, client=client)

        # intialize round
        status.FL_round = 1

        fl_start_time = time.time()  # FL Client start time

        await loop.run_in_executor(None, request)  # play asynchronously FL Client 

        logging.info('fl learning finished')

        fl_end_time = time.time() - fl_start_time  # FL Client end time

        client_all_time_result = {"client_num": status.FL_client_num, "operation_time": fl_end_time}
        json_all_time_result = json.dumps(client_all_time_result)
        logging.info(f'client_operation_time - {json_all_time_result}')


        # delete client & fl_client_start request
        del client, request

        # Complete Client learning 
        await utils.notify_fin()
        logging.info('FL Client Learning Finish')

    except Exception as e:
        logging.info('[E][PC0002] learning', e)
        status.FL_client_fail = True
        await utils.notify_fail()
        status.FL_client_fail = False
        raise e


# load dataset
def load_partition():
    # Load the dataset partitions
    global status

    # Cifar 10 데이터셋 불러오기
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # client_num 값으로 데이터셋 나누기
    (X_train, y_train) = X_train[status.FL_client_num * 2000:(status.FL_client_num + 1) * 2000], y_train[
                                                                           status.FL_client_num * 2000:(status.FL_client_num + 1) * 2000]
    (X_test, y_test) = X_test[status.FL_client_num * 1000:(status.FL_client_num + 1) * 1000], y_test[status.FL_client_num * 1000:(status.FL_client_num + 1) * 1000]

    # class 설정
    num_classes = 10

    # one-hot encoding class 범위 지정
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(y_train, num_classes)
    test_labels = to_categorical(y_test, num_classes)

    # 전처리
    train_features = X_train.astype('float32') / 255.0
    test_features = X_test.astype('float32') / 255.0


    # data check => IID VS Non IID
    # array -> list
    y_list = y_train.tolist()
    y_train_label = list(itertools.chain(*y_list))
    counter = Counter(y_train_label)

    # check client data(label) => non-iid
    for i in range(num_classes):
        data_check_dict = {"client_num": int(status.FL_client_num), "label_num": i, "data_size": int(counter[i])}
        data_check_json = json.dumps(data_check_dict)
        logging.info(f'data_check - {data_check_json}')


    return (train_features, train_labels), (test_features, test_labels)


if __name__ == "__main__":

    try:
        # create client api => to connect client manager
        uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)
        
    finally:
        # FL client out
        requests.get('http://localhost:8003/flclient_out')
        logging.info('%s client close'%client_num)
