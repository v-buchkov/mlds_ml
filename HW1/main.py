from typing import List, Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder

import pickle

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

app = FastAPI()

templates = Jinja2Templates(directory="templates")


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def preprocess_data(car: pd.DataFrame) -> pd.DataFrame:
    car['mileage'] = car['mileage'].apply(
        lambda x: float(x.rstrip(' kmpl').rstrip(' km/kg')) if type(x) != float else x)
    car['engine'] = car['engine'].apply(lambda x: float(x.rstrip(' CC')) if type(x) != float else x)

    car['max_power'] = car['max_power'].apply(lambda x: 0.0 if x == ' bhp' else x)
    car['max_power'] = car['max_power'].apply(lambda x: float(x.rstrip(' bhp')) if type(x) != float else x)

    # Do all the same for the test
    car['torque'] = car['torque'].astype(str)
    car['torque'] = car['torque'].apply(
        lambda x: x.split('@')[0] + x.split('(')[1].split('@')[0] + '@' + x.split('@')[1].split('(')[
            0] if '(' in x else x)
    car['torque'] = car['torque'].apply(lambda x: x.replace('at', '@').replace('/', '@'))
    car[['torque', 'max_torque_rpm']] = car['torque'].str.split('@', 1, expand=True)

    NM_KGM_CONSTANT = 9.80665
    car['torque'] = car['torque'].apply(lambda x: x.replace(' ', '').upper().split('(')[0])
    car['torque'] = car['torque'].apply(
        lambda x: float(x.rstrip('NM')) if 'NM' in x else float(x.rstrip('KGM')) * NM_KGM_CONSTANT)
    car['max_torque_rpm'] = car['max_torque_rpm'].apply(
        lambda x: str(x).replace(',', '').replace(' ', '').split('-')[0].split('~')[0].split('+')[0].upper().rstrip(
            'RPM'))
    car['max_torque_rpm'] = car['max_torque_rpm'].apply(lambda x: float(x) if x != 'NONE' else None)
    car['max_torque_rpm'] = car['torque'] / car['max_torque_rpm']

    car['engine'] = car['engine'].astype(int)

    car['seats'] = car['seats'].astype(int)

    X_test_cat = car.drop(['name'], axis=1, inplace=False)

    for c in ['fuel', 'seller_type', 'transmission', 'owner']:
        ohe = pickle.load(open(f'pre_trained_data/ohe_{c}.pkl', 'rb'))
        transformed_test = ohe.transform(X_test_cat[[c]])
        X_test_cat[ohe.categories_[0]] = transformed_test.toarray()
        X_test_cat.drop([c], axis=1, inplace=True)

    X_test_cat.drop(['Diesel', 'Dealer', 'Automatic', 'First Owner'], axis=1, inplace=True)

    # Уберём max_torque_rpm - потенциальна мультиколлинеарность с torque
    X_test_cat.drop(['max_torque_rpm'], axis=1, inplace=True)

    # Добавим логарифмический признак для year
    X_test_cat['year_log'] = X_test_cat['year'].apply(lambda x: np.log(x))

    # Добавим "износ вождения" - km_driven / years (чтобы учитывать машины, которые стояли в гараже, но на них не ездили)
    X_test_cat['exhaustion'] = X_test_cat['km_driven'] / X_test_cat['year']

    # Добавим силу на мощность мотора
    X_test_cat['capacity'] = X_test_cat['max_power'] / X_test_cat['engine']

    # Из названия авто возьмём название марки авто, примением OneHotEncoder, чтобы категоризовать
    X_test_cat['brand'] = car['name'].apply(lambda x: x.split(' ')[0])

    ohe = pickle.load(open(f'pre_trained_data/ohe_brand.pkl', 'rb'))
    transformed_test = ohe.transform(X_test_cat[['brand']])
    X_test_cat[ohe.categories_[0]] = transformed_test.toarray()
    X_test_cat.drop(['brand'], axis=1, inplace=True)

    X_test_cat.drop(['Nissan'], axis=1, inplace=True)

    return X_test_cat


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    input_data = pd.DataFrame(jsonable_encoder(item), index=[0])
    pickled_model = pickle.load(open('pre_trained_data/model_data.pkl', 'rb'))

    X = preprocess_data(input_data)
    predicted_price = pickled_model.predict(X)[0]
    return round(predicted_price, 2)


@app.post("/predict_items")
async def predict_items(items: Items) -> List[Any]:
    input_data = pd.DataFrame(items)
    pickled_model = pickle.load(open('pre_trained_data/model_data.pkl', 'rb'))
    X = preprocess_data(input_data)
    input_data['selling_price'] = pickled_model.predict(X)[0]
    return input_data.tolist()
