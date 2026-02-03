import math
import os
import threading
import traceback
from sched import scheduler

import joblib
import plotly
import schedule
import time
import json
import glob
import random

import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from pandas import to_datetime

from bottle import Bottle, template, run, static_file, HTTPError, request, response
from datetime import datetime, timedelta

from logging_config import setup_logger
from collections import OrderedDict

from forecast import Forecaster as Forecast
import forecast.ForecasterManager as ForecasterManager
import forecast.OptimalScheduler as OptimalScheduler
import sqlDB as db
import blockchain as Blockchain
import numpy as np


# LOGGER COLORS
logger = setup_logger()

# Helper function per convertir tipus NumPy/Pandas a tipus natius de Python
def convert_to_json_serializable(obj):
    """
    Converteix recursivament objectes amb tipus NumPy/Pandas a tipus natius de Python
    per permetre la serialització JSON.
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# PARÀMETRES DE L'EXECUCIÓ
HOSTNAME = '0.0.0.0'
PORT = 55023


#INICIACIÓ DE L'APLICACIÓ I LA BASE DE DADES
app = Bottle()
database = db.SqlDB()
forecast = Forecast.Forecaster(debug=True)
optimalScheduler = OptimalScheduler.OptimalScheduler(database)
blockchain = Blockchain.Blockchain()

# Ruta per servir fitxers estàtics i imatges des de 'www'
@app.get('/static/<filepath:path>')

#region HTML PAGES

#region core_paths
def serve_static(filepath):
    return static_file(filepath, root='./images/')

@app.get('/resources/<filepath:path>')
def serve_resources(filepath):
    return static_file(filepath, root='./resources/')

@app.get('models/<filepath:path>')
def serve_models(filepath):
    return static_file(filepath, root='./models/')

# Ruta dinàmica per a les pàgines HTML
@app.get('/<page>')
def get_page(page):
    # Ruta del fitxer HTML
    # file_path = f'./www/{page}.html'
    # Comprova si el fitxer existeix.
    if os.path.exists(f'./www/{page}.html'):
        # Control de dades segons la pàgina
        return static_file(f'{page}.html', root='./www/')
    elif os.path.exists(f'./www/{page}.css'):
        return static_file(f'{page}.css', root='./www/')
    else:
        return HTTPError(404, "La pàgina no existeix")

#endregion core_paths

# Ruta inicial
@app.get('/')
def get_init():
    ip = request.environ.get('REMOTE_ADDR')
    token = database.supervisor_token


    aux = database.get_forecasts_name()
    active_sensors = [x[0] for x in aux]

    return template('./www/main.html',
                    ip = ip,
                    token = token,
                    active_sensors_list = active_sensors)

@app.get('/sensors')
def sensors_page():
    return template('./www/sensors.html',)

@app.get('/databaseView')
def database_graph_page():
    sensors_id = database.get_all_saved_sensors_id()
    graphs_html = {}

    return template('./www/databaseView.html', sensors_id=sensors_id, graphs=graphs_html)

@app.get('/model')
def create_model_page(active_model = "None"):
    try:
        sensors_id = database.get_all_saved_sensors_id()
        models_saved = [os.path.basename(f)
                        for f in glob.glob(forecast.models_filepath + "forecastings/*.pkl")]

        forecasts_aux = database.get_forecasts_name()
        forecasts_id = []
        for f in forecasts_aux:
            forecasts_id.append(f[0])

        if active_model == "None": active_model = "newModel"


        return template('./www/model.html',
                        sensors_input = sensors_id,
                        models_input = models_saved,
                        forecasts_id = forecasts_id,
                        active_model = active_model)
    except Exception as ex:
        error_message = traceback.format_exc()
        return f"Error! Alguna cosa ha anat malament :c : {str(ex)}\nFull Traceback:\n{error_message}"

@app.route('/config_page')
def config_page():

    sensors_id = database.get_all_saved_sensors_id(kw=True)
    user_lat = optimalScheduler.latitude
    user_long = optimalScheduler.longitude
    user_location = {'lat': user_lat, 'lon': user_long}

    user_data = get_user_configuration_data()

    return template('./www/config_page.html',
                    sensors = sensors_id,
                    location = user_location,
                    user_data = user_data)

@app.route('/optimization')
def optimization_page():

    # RESTRICCIONS PER A DISPOSITIU
    config_path = 'resources/optimization_devices.conf'
    devices_data = {}

    if not os.path.exists(config_path):
        logger.warning(f"⚠️ - No s'ha trobat el fitxer de configuració: {config_path}")
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            devices_data = json.load(f)

    # DISPOSITIUS I ENTITATS ASSOCIADES
    devices_entities = database.get_devices_info()

    current_date = datetime.now().strftime('%d-%m-%Y')
    return template("./www/optimization.html",
                    current_date = current_date,
                    device_types = json.dumps(devices_data),
                    device_entities = devices_entities)

#endregion PAGE CREATIONS

#region PÀGINA MAIN
@app.route('/get_scheduler_data')
def get_scheduler_data():
    try:
        today = datetime.today().strftime("%d_%m_%Y")
        full_path = os.path.join(forecast.models_filepath, "optimizations/"+today+".pkl")
        if not os.path.exists(full_path):
            optimize(today=True)

        if not os.path.exists(full_path): return json.dumps("ERROR")

        optimization_db = joblib.load(full_path)

        graph_timestamps = optimization_db['timestamps']
        graph_optimization = optimization_db['total_balance']


        graph_df = pd.DataFrame({
            "hora": pd.to_datetime(graph_timestamps),
            "optimitzacio": graph_optimization,
            # "consum": graph_consum,
            # "generacio": graph_generation
        })
        graph_df['hora_str'] = graph_df['hora'].dt.strftime('%H:%M')

        fig = go.Figure()

        # Línia principal (verd amb fill)
        fig.add_trace(go.Scatter(
            x=graph_df["hora"],
            y=graph_df["optimitzacio"],
            mode='lines',
            name="Optimització",
            line=dict(color="green", width=2),
            fill='tozeroy',
            fillcolor="rgba(0,128,0,0.3)"
        ))

        now = datetime.now()

        fig.update_layout(
            height=400,
            yaxis=dict(zeroline=False),
            xaxis=dict(
                tickmode='array',
                tickvals=graph_timestamps,
                ticktext=graph_df['hora_str'],
                tickangle=-45,
                tickfont=dict(size=13),
                title="Hores"
            ),
            yaxis_title="Consum (W)",
            shapes=[
                dict(
                    type="line",
                    x0=now,
                    x1=now,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="red", width=2, dash="dash")
                )
            ],
            annotations=[
                dict(
                    x=now,
                    y=1.2,                 # una mica per sobre del gràfic
                    xref="x",
                    yref="paper",
                    text="Actual",
                    showarrow=False,
                    font=dict(color="red", size=12),
                    textangle=-45            # rotat en diagonal
                )
            ],
        )

        fig_json = fig.to_plotly_json()
        response.content_type = "application/json"
        return json.dumps(fig_json, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        logger.exception(f"❌ Error obtenint scheduler': {e}")
#endregion PÀGINA MAIN

#region PÀGINA DEVICES

@app.get('/get_sensors')
def get_sensors():
    try:
        all_devices_data = database.get_all_sensors_data()

        response.content_type = 'application/json'
        return json.dumps(all_devices_data)

    except Exception as ex:
        error_message = traceback.format_exc()
        return f"Error! Alguna cosa ha anat malament :c : {str(ex)}\nFull Traceback:\n{error_message}"

@app.route('/update_sensors', method='POST')
def update_sensors():
    data = request.json
    if not data:
        response.status = 400
        return {"status":"error", "msg": "Dades buides"}

    database.reset_all_sensors_save()

    for device in data:
        database.update_sensor_active(sensor = device['entityId'], active = True)
        database.update_sensor_type(sensor = device['entityId'], new_type = device['type'])

    return {"status": "ok", "msg": f"Sensors guardats"}

@app.route('/self_destruct', method='POST')
def self_destruct_database():
    database.self_destruct()
    return {"status": "ok"}

#endregion PÀGINA DEVICES

#region PÀGINA DATABASE

@app.route('/get_graph_info', method='POST')
def graphs_view():
    try:
        selected_sensors = request.forms.get("sensors_id")
        selected_sensors_list = [sensor.strip() for sensor in selected_sensors.split(',')] if selected_sensors else []
        date_to_check = ''

        date_to_check_input = request.forms.getall("datetimes")
        if  not date_to_check_input:
            # Mostrar per defecte els últims 14 dies (abans era de 30)
            start_date = datetime.today() - timedelta(days=14)
            end_date = datetime.today()
            date_label = None
        else:
            date_to_check = date_to_check_input[0].split(' - ')
            start_date = datetime.strptime(date_to_check[0], '%d/%m/%Y %H:%M').strftime("%Y-%m-%dT%H:%M:%S") + '+00:00'
            end_date = datetime.strptime(date_to_check[1], '%d/%m/%Y %H:%M').strftime("%Y-%m-%dT%H:%M:%S") + '+00:00'
            date_label = date_to_check_input[0]


        sensors_data = database.get_all_saved_sensors_data(selected_sensors_list, start_date, end_date)

        response = {
            "status": "ok",
            "range":{
                "start": start_date,
                "end": end_date,
                "label": date_label,
            },
            "graphs": {}
        }

        if len(sensors_data) == 0:
            return json.dumps({
                "status": "empty",
                "message": "No hi ha dades disponibles",
                "graphs": {}
            })

        for sensor_id, records in sensors_data.items():
            timestamps = []
            values = []

            for ts, value in records:
                if value is not None:
                    timestamps.append(ts)
                    values.append(value)

            if not values:
                response["graphs"][sensor_id] = {
                    "status": "no-data",
                    "message": f"No hi ha dades del sensor {sensor_id}"
                }
                continue

            response["graphs"][sensor_id] = {
                "status": "ok",
                "timestamps": timestamps,
                "values": values
            }

        return json.dumps(response)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@app.route('/force_update_database')
def force_update_database():
    database.update_database("all")
    database.clean_database_hourly_average(all_sensors=True)
    return "ok"

#endregion PÀGINA DATABASE

#region PÀGINA MODEL

@app.route('/get_model_config/<model_name>')
def get_model_config(model_name):
    try:
        model_path = os.path.join(forecast.models_filepath,'forecastings/',f"{model_name}.pkl")
        config = dict()
        with open(model_path, 'rb') as f:
            config = joblib.load(f)

        response_config = ""
        response_config += f"algorithm = {config.get('algorithm','')}\n"
        response_config += f"scaler = {config.get('scaler_name','')}\n"
        response_config += f"sensorsId = {config.get('sensors_id','')}\n"
        response_config += f"meteo_data = {str(config.get('meteo_data_is_selected', False)).lower()}\n"

        extra = config.get('extra_sensors', {})
        extra_sensors_id = ",".join(extra.keys()) if isinstance(extra, dict) else ''
        response_config += f"extra_sensors = {extra_sensors_id}\n"

        if "params" in config:
            for k, v in config["params"].items():
                if k == 'bootstrap':
                    aux = 'true' if v else 'false'
                    response_config += f"{k} = {aux}\n"
                elif k == 'algorithm':
                    response_config += f"KNN_algorithm = {v}\n"
                else:
                    response_config += f"{k} = {v}\n"
        if "max_time" in config:
            response_config += f"max_time = {config['max_time']}\n"

        return response_config

    except Exception as e:
        return f"Error! : {str(e)}"

@app.route('/get_model_metrics/<model_name>')
def get_model_metrics(model_name):
    """
    Retorna les mètriques d'un model guardat
    """
    try:
        model_path = os.path.join(forecast.models_filepath,'forecastings/',f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return json.dumps({"status": "error", "message": "Model not found"})
        
        with open(model_path, 'rb') as f:
            model_db = joblib.load(f)
        
        metrics = model_db.get('metrics', {})
        train_val_test = model_db.get('train_val_test_split', {})
        
        # Convertir tipus NumPy/Pandas a tipus natius de Python
        metrics = convert_to_json_serializable(metrics)
        train_val_test = convert_to_json_serializable(train_val_test)
        
        response = {
            "status": "ok",
            "metrics": metrics,
            "train_val_test_split": train_val_test
        }
        
        return json.dumps(response)
    
    except Exception as e:
        logger.error(f"❌ Error getting metrics for model {model_name}: {e}")
        return json.dumps({"status": "error", "message": str(e)})

def train_model():
    selected_model = request.forms.get("model")
    extra_sensors_id = request.forms.get("sensors_id") if request.forms.get("sensors_id") else None
    config = {}

    for key in request.forms.keys():
        if key != "model" or key != "sensors_id" or key != 'action':
            value = request.forms.get(key)
            value = value.strip().lower()

            if value in ["true", "false", "null", "none"]:
                if value == "true": config[key] = True
                elif value == "false": config[key] = False
                else: config[key] = None
            elif value.isdigit():
                config[key] = int(value)
            else:
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value

    sensors_id = config.get("sensorsId")
    scaled = config.get("scaled")
    model_name = config.get("modelName")
    
    # Obtenir configuració de windowing
    windowing_option = config.get("windowingOption", "default")
    look_back = None
    
    if windowing_option == "24-48":
        look_back = {-1: [24, 48]}
    elif windowing_option == "48-72":
        look_back = {-1: [48, 72]}
    elif windowing_option == "1-24":
        look_back = {-1: [1, 24]}
    elif windowing_option == "custom":
        window_start = config.get("windowStart", 25)
        window_end = config.get("windowEnd", 48)
        look_back = {-1: [window_start, window_end]}
    # Si és "default" o None, es farà servir el valor per defecte {-1: [25, 48]}

    config.pop("sensorsId")
    config.pop("scaled")
    config.pop("modelName")
    config.pop('model')
    config.pop("models")
    config.pop("action")
    if 'sensors_id' in config: config.pop('sensors_id')
    if 'windowingOption' in config: config.pop('windowingOption')
    if 'windowStart' in config: config.pop('windowStart')
    if 'windowEnd' in config: config.pop('windowEnd')

    if "meteoData" in config:
        meteo_data = True
        config.pop("meteoData")
    else:
        meteo_data = False

    if model_name == "":
        aux = sensors_id.split('.')
        model_name = aux[1]
    if scaled == 'None': scaled = None

    # Filtrar dades d'entrenament als últims 14 dies
    #cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=14)
    
    extra_sensors_df = {}
    if extra_sensors_id is None:
        extra_sensors_id = None
    elif len(extra_sensors_id) == 1 and extra_sensors_id[0] == "None":
        extra_sensors_id = None
    else:
        if "None" in extra_sensors_id: extra_sensors_id.remove('None')
        extra_sensors_df = {}
        extra_sensors_list = [s.strip() for s in extra_sensors_id.split(',') if s.strip()]
        for s in extra_sensors_list:
            aux = database.get_data_from_sensor(s)
            # Filtrar dades
            if not aux.empty and 'timestamp' in aux.columns:
                aux['timestamp'] = pd.to_datetime(aux['timestamp'])
                if aux['timestamp'].dt.tz is None:
                     aux['timestamp'] = aux['timestamp'].dt.tz_localize('UTC')
                #aux = aux[aux['timestamp'] >= cutoff_date]
            extra_sensors_df[s] = aux


    sensors_df = database.get_data_from_sensor(sensors_id)
    # Filtrar dades
    if not sensors_df.empty and 'timestamp' in sensors_df.columns:
        sensors_df['timestamp'] = pd.to_datetime(sensors_df['timestamp'])
        if sensors_df['timestamp'].dt.tz is None:
             sensors_df['timestamp'] = sensors_df['timestamp'].dt.tz_localize('UTC')
        #sensors_df = sensors_df[sensors_df['timestamp'] >= cutoff_date]

    logger.info(f"Selected model: {selected_model}, Config: {config}, Windowing: {look_back}")

    lat = optimalScheduler.latitude
    lon = optimalScheduler.longitude

    if selected_model == "AUTO":
        forecast.create_model(data=sensors_df,
                              sensors_id=sensors_id,
                              y='value',
                              escalat=scaled,
                              max_time=config['max_time'],
                              filename=model_name,
                              meteo_data= meteo_data if meteo_data is True else None,
                              extra_sensors_df=extra_sensors_df if extra_sensors_id is not None else None,
                              lat=lat,
                              lon=lon,
                              look_back=look_back)
    else:
        forecast.create_model(data=sensors_df,
                              sensors_id=sensors_id,
                              y='value',
                              algorithm=selected_model,
                              params=config,
                              escalat=scaled,
                              filename=model_name,
                              meteo_data=meteo_data if meteo_data is True else None,
                              extra_sensors_df= extra_sensors_df if extra_sensors_id is not None else None,
                              lat=lat,
                              lon=lon,
                              look_back=look_back)

    return model_name

def forecast_model(selected_forecast):
    forecast_df, real_values, sensor_id = ForecasterManager.predict_consumption_production(model_name=selected_forecast)

    forecasted_done_time = datetime.today().strftime('%d-%m-%Y')
    timestamps = forecast_df.index.tolist()
    predictions = forecast_df['value'].tolist()

    rows = []
    
    # LIMITAR EL GRÀFIC: Només guardem les dades dels últims 14 dies (i futur)
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=14)
    cutoff_date = cutoff_date.replace(tzinfo=None) # Comparació en 'naive' (sense zona horària) per evitar errors
    
    for i in range(len(timestamps)):
        ts = timestamps[i]
        # Normalitzem a naive per comparar
        ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts

        if ts_naive >= cutoff_date:
            forecasted_time = ts.strftime("%Y-%m-%d %H:%M")
            predicted = predictions[i]
            actual = real_values[i] if i < len(real_values) else None
    
            rows.append((selected_forecast, sensor_id, forecasted_done_time, forecasted_time, predicted, actual))
            
    logger.info(f"📈 Forecast realitzat correctament")
    database.save_forecast(rows)

def delete_model():
    selected_model = request.forms.get("models")
    database.remove_forecast(selected_model)

    model_path = forecast.models_filepath +'forecastings/'+ selected_model
    if os.path.exists(model_path):
        os.remove(model_path)
        logger.info(f"Model deleted: {model_path}")
    else:
        logger.error(f"Model {selected_model} not found")

@app.route('/submit-model', method="POST")
def submit_model():
    try:
        action = request.forms.get('action')

        if action == 'train':
            model_name = train_model()
            return create_model_page(model_name)
        elif action == 'forecast':
            selected_forecast = request.forms.get("models")
            forecast_model(selected_forecast)
            forecast_without_suffix = selected_forecast.replace('.pkl', '')
            return create_model_page(forecast_without_suffix)
        elif action == 'delete':
            delete_model()
            return create_model_page()


    except Exception as e:
        error_message = traceback.format_exc()
        return f"Error! The model could not be processed : {str(e)}\nFull Traceback:\n{error_message}"

@app.route('/get_forecast_data/<model_name>')
def get_forecast_data(model_name):
    try:
        today_date = datetime.today().strftime('%d-%m-%Y')
        yesterday_date = (datetime.today() - timedelta(days = 1)).strftime('%d-%m-%Y')

        forecasts = database.get_data_from_forecast_from_date(model_name + ".pkl", today_date)
        yesterday = database.get_data_from_forecast_from_date(model_name + ".pkl", yesterday_date)

        if forecasts.empty:
            predictions = ""
            timestamps = ""
        else:
            timestamps = forecasts["timestamp"].tolist()
            predictions = forecasts["value"].tolist()

        if yesterday.empty:
            yesterday_predictions = ""
            yesterday_timestamps = ""
        else:
            yesterday_predictions = yesterday["value"].tolist()
            yesterday_timestamps = yesterday["timestamp"].tolist()



        real_values = []
        real_values_timestamps = []
        for i in range(len(forecasts['real_value'])):
            if not math.isnan(forecasts['real_value'][i]):
                real_values.append(forecasts['real_value'][i])
                real_values_timestamps.append(forecasts['timestamp'].tolist()[i])

        start_timestamp = (datetime.today() - timedelta(days=4)).replace(hour=0, minute=0).strftime('%Y-%m-%d %H:%M')
        last_timestamp = (datetime.today() + timedelta(days=4)).replace(hour=0, minute=0).strftime('%Y-%m-%d %H:%M')

        return json.dumps({
            "status": "ok",
            "timestamps": timestamps,
            "predictions": predictions,
            "real_values": real_values,
            "real_values_timestamps": real_values_timestamps,
            "yesterday_predictions": yesterday_predictions,
            "yesterday_timestamps": yesterday_timestamps,
            "start_timestamp": start_timestamp,
            "last_timestamp": last_timestamp,
        })


    except Exception as e:
        logger.error(f"❌ Error getting forecast for model {model_name}: {e}")
        return json.dumps({"status": "error", "message": str(e)})

#endregion PÀGINA MODEL

#region PÀGINA CONFIGURACIÓ
@app.post('/save_config')
def save_config():
    try:

        data = request.json
        consumption = data.get('consumption')
        generation = data.get('generation')
        name = data.get('name')


        config_dir = forecast.models_filepath + 'config/user.config'
        os.makedirs(forecast.models_filepath + 'config', exist_ok=True)

        database.update_sensor_active(sensor=consumption, active=True)
        database.update_sensor_active(sensor=generation, active=True)

        numero_entero = random.randint(0, 9999999999)
        claves = blockchain.generar_claves_ethereum(f"esta es mi frase secreta para generar claves {numero_entero}")
        logger.info(f"Clau privada: {claves['private_key']}")
        logger.info(f"Dirección Ethereum : {claves['public_key']}")
        res_add_user = blockchain.registrar_usuario(claves['public_key'], claves['private_key'])
        logger.debug(f"res_add_user: {res_add_user}")




        joblib.dump({ 'consumption': consumption,
                            'generation': generation,
                            'name' : name,
                            'public_key': claves['public_key'],
                            'private_key': claves['private_key']}, config_dir)

        logger.info(f"Configuració guardada al fitxer {config_dir}")

        certificate_hourly_task()
        return "OK"

    except Exception as e:
        logger.error(f"Error saving config file :c : {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error! : {str(e)}"

@app.route('/delete_config', method='DELETE')
def delete_config():
    user_config_path = forecast.models_filepath + '/config/user.config'
    if os.path.exists(user_config_path):
        aux = joblib.load(user_config_path)
        consumption = aux['consumption']
        generation = aux['generation']
        database.update_sensor_active(sensor=consumption, active=False)
        database.update_sensor_active(sensor=generation, active=False)

        os.remove(user_config_path)
        return 'Config file deleted successfully'
    else:
        return 'Config file not found'

@app.route('/get_res_certify_data')
def get_res_certify_data():
    try:
        full_path = os.path.join(forecast.models_filepath, "config", "res_certify.pkl")
        status = "ERROR"
        data = {}

        if os.path.exists(full_path):
            data = joblib.load(full_path)
            status = "OK"

        return json.dumps({
            "status": status,
            "data": data
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return f"Error! : {str(e)}"

def get_user_configuration_data():
    config_dir = forecast.models_filepath + 'config/user.config'
    user_data = {
        'name': '',
        'consumption': '',
        'generation': '',
        'locked': False,
    }

    if os.path.exists(config_dir):
        aux = joblib.load(config_dir)
        user_data['name'] = aux['name']
        user_data['consumption'] = aux['consumption']
        user_data['generation'] = aux['generation']
        user_data['locked'] = True

    return user_data

#endregion PÀGINA CONFIGURACIÓ

#region PÀGINA OPTIMITZACIÓ

@app.route('/run_optimization')
def run_optimization():
    optimize(today=True)

@app.route('/optimize')
def optimize(today = False):
    try:
        horizon = 24
        horizon_min = 1 # 1 = 60 minuts  | 2 = 30 minuts | 4 = 15 minuts

        user_data = get_user_configuration_data()
        global_consumer_id = user_data['consumption']
        global_generator_id = user_data['generation']

        if global_generator_id == '' or global_consumer_id == '':
            logger.warning("⚠️ Variables globals no seleccionades a la configuració d'usuari.")
            return 'ERROR'

        price = []

        success, devices_config, price, total_balance_hourly = optimalScheduler.start_optimization(
            consumer_id = global_consumer_id,
            generator_id = global_generator_id,
            horizon = horizon,
            horizon_min = horizon_min)

        if success:
            # GUARDAR A FITXER
            optimization_result = {
                "timestamps": optimalScheduler.timestamps,
                "total_balance": total_balance_hourly,
                "total_price": price,
                "devices_config": devices_config
            }
            if today:
                save_date = datetime.today().strftime("%d_%m_%Y")
            else:
                save_date = (datetime.today() + timedelta(days=1)).strftime("%d_%m_%Y")
            full_path = os.path.join(forecast.models_filepath, "optimizations/"+save_date+".pkl")
            os.makedirs(forecast.models_filepath + 'optimizations', exist_ok=True)
            if os.path.exists(full_path):
                logger.warning("Eliminant arxiu antic d'optimització ")
                os.remove(full_path)

            joblib.dump(optimization_result, full_path)
            logger.info(f"✏️ Optimització diària guardada al fitxer {full_path}")

            schedule.clear('device_config_tasks')
            schedule.every().hour.at(":00").do(config_optimized_devices_HA).tag('device_config_tasks')
            logger.info("📅 Job programat per executar-se un cop cada hora (als minuts :00)")

    except Exception as e:
        logger.error(f"❌ Error optimitzant: {str(e)}: {traceback.format_exc()}")

def flexibility():
    """
    Calcula la flexibilitat de l'optimització realitzada dins OptimalScheduler.SolucioFinal
    """

    full_path = os.path.join(forecast.models_filepath, "optimizations/sonnen_opt.pkl")

    # if not os.path.exists(full_path): optimize()

    sonnen_db = joblib.load(full_path)

    SoC_max = 25 # Capacitat màxima de la bateria
    SoC_min = 0  # Capacitat mínima per protegir la bateria
    Pc_max = 2.5  # Potència màxima de la bateria Kw  (especificat a la bateria)
    Pd_max = 2.5  # Potència màxima de descàrrega Kw (especificat a la bateria)
    eff = 0.95   # Eficiència de càrrega
    delta_t = 1  # Interval horari (hora)

    fup = []
    fdown = []

    for t in range(len(sonnen_db['timestamps'])):
        SoC_t = sonnen_db['SoC'][t]  # Estat de càrrega de la bateria a hora T
        Pb_t = sonnen_db['Power'][t]     # Potència actual de la bateria

        flex_up = max(0,
                       min(Pc_max,
                            (SoC_max - SoC_t) / (eff * delta_t)) - Pb_t)

        flex_down = max(0,
                        Pb_t + min(Pd_max,
                                   SoC_t - SoC_min) / delta_t)

        fup.append(flex_up)
        fdown.append(flex_down)

    return fup, fdown, sonnen_db['Power'], sonnen_db['timestamps']

def generate_plotly_flexibility():
    Fup, Fdown, consum, timestamps = flexibility()

    fup_line = [consum[t] + Fup[t] for t in range(len(timestamps))]
    fdown_line = [consum[t] - Fdown[t] for t in range(len(timestamps))]

    graph_df = pd.DataFrame({
        "hora": pd.to_datetime(timestamps),
        "optimitzacio": consum,
        "Fup": fup_line,
        "Fdown": fdown_line
    })

    graph_long = graph_df.melt(
        id_vars=["hora"],  # columna que es manté tal qual
        value_vars=["optimitzacio", "Fdown", "Fup"],  # columnes que es transformen
        var_name="Serie",  # nom de la columna nova per als noms de les sèries
        value_name="Valor"  # nom dels valors
    )

    fig = px.line(
        graph_long,
        x="hora",
        y="Valor",
        color="Serie",
        markers=True,
        title="Gràfica de Flexibilitat"
    )

    fig.update_xaxes(dtick=3600000)  # 3600000 ms = 1 hora
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_xaxes(tickangle=45)

@app.post('/get_config_file_names')
def get_config_file_names():
    # CONFIGURACIONS CREADES
    created_configs_path = forecast.models_filepath + "/optimizations/configs"

    if os.path.exists(created_configs_path) and os.path.isdir(created_configs_path):
        json_config_files = [f for f in os.listdir(created_configs_path) if f.endswith(".json")]

        if len(json_config_files) == 0: return {"status": "error"}
        return {"status": "ok", "names" : json_config_files}
    else:
        return {"status": "error"}

@app.post('/save_optimization_config')
def save_optimization_config():
    data = request.json
    if not data:
        response.status = 400
        return {"status":"error", "msg": "Dades buides"}

    device_name = data.get("device_name")

    full_path = os.path.join(forecast.models_filepath, "optimizations/configs/"+ device_name +".json")
    os.makedirs(forecast.models_filepath + 'optimizations/configs', exist_ok=True)

    if os.path.exists(full_path):
        logger.warning(f"Eliminant arxiu antic de configuració {device_name}")
        os.remove(full_path)

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    for var in data['extra_vars'].values():
        database.update_sensor_active(var['sensor_id'], True)


    return {"status": "ok", "msg": f"Optimització desada com {device_name}.json"}

@app.post('/delete_optimization_config/<file_name>')
def delete_optimization_config(file_name):
    logger.debug(f"eliminant info {file_name}")
    full_path = os.path.join(forecast.models_filepath, "optimizations/configs/" + file_name)
    logger.debug(full_path)
    if os.path.exists(full_path):
        os.remove(full_path)
        return {"status": "ok"}
    else:
        return {"status": "error", "msg": "No existeix arxiu {file_name}.json"}

@app.route('/get_device_config_data/<file_name>')
def get_device_config_data(file_name):
    config_path = forecast.models_filepath + "/optimizations/configs/" + file_name
    device_config = {}

    if not os.path.exists(config_path):
        response.status = 400
        return {"status": "error", "msg": "Dades buides"}

    with open(config_path, 'r', encoding='utf-8') as f:
        device_config = json.load(f)
    
    today = datetime.today().strftime("%d_%m_%Y")
    device_config_path = os.path.join(forecast.models_filepath, "optimizations/" + today + ".pkl")
    if not os.path.exists(device_config_path):
        return {"status": "ok", "device_config": device_config}
    optimization_db = joblib.load(device_config_path)
    fixed_name = file_name.removesuffix(".json")
    device_config['hourly_config'] = optimization_db['devices_config'][fixed_name].tolist()
    device_config['timestamps'] = pd.to_datetime(optimization_db['timestamps']).strftime('%Hh').tolist()

    return {"status": "ok", "device_config": device_config}

#endregion PÀGINA OPTIMITZACIÓ

#region DAILY TASKS

def daily_task():
    
    try:
        # Actualitzem la base de dades
        hora_actual = datetime.now().strftime('%Y-%m-%d %H:00')
        database.update_database("all")
        database.clean_database_hourly_average(all_sensors=True)

        # Optimització
        logger.warning(f"📈 [{hora_actual}] - INICIANT PROCÉS D'OPTIMITZACIÓ")
        optimize(today=False)
    except Exception as e:
        hora_actual = datetime.now().strftime('%Y-%m-%d %H:00')
        logger.error(f" ❌ [{hora_actual}] - ERROR al daily task : {e}")

def monthly_task():
    try:
        # Eliminació de dades sense cap activitat
        today = datetime.today()
        last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1) #últim dia del mes
        if today == last_day:
            sensors_id = database.get_all_sensors()
            sensors_id = sensors_id['entity_id'].tolist()

            for sensor_id in sensors_id:
                is_active = database.get_sensor_active(sensor_id)
                if not is_active:
                    database.remove_sensor_data(sensor_id)

            logger.debug(f"Running monthly task at {datetime.now().strftime('%d-%b-%Y   %X')}" )
            
    except Exception as e:
        hora_actual = datetime.now().strftime('%Y-%m-%d %H:00')
        logger.error(f" ❌ [{hora_actual}] - ERROR al monthly task : {e}")

def daily_forecast_task():
    try:
        hora_actual = datetime.now().strftime('%Y-%m-%d %H:00')
        database.update_database("all")
        logger.debug(f"📈 [{hora_actual}] - STARTING DAILY FORECASTING")
        models_saved = [os.path.basename(f) for f in glob.glob(forecast.models_filepath + "forecastings/*.pkl")]
        for model in models_saved:
            model_path = os.path.join(forecast.models_filepath, "forecastings/" ,f"{model}")
            with open(model_path, 'rb') as f:
                config = joblib.load(f)
            aux = config.get('algorithm','')
            if aux != '':
                # daily_train_model(config, model)
                logger.debug(f"     Running daily forecast for {model}")
                forecast_model(model)
        logger.debug("ENDING DAILY FORECASTS")
        
    except Exception as e:
        hora_actual = datetime.now().strftime('%Y-%m-%d %H:00')
        logger.error(f" ❌ [{hora_actual}] - ERROR al daily forecast : {e}")

def certificate_hourly_task():
    try:
        logger.info(f"🕒 Running certificate hourly task at {datetime.now().strftime('%H:%M')}")
        config_dir = forecast.models_filepath + 'config/user.config'
        if os.path.exists(config_dir):
            aux = joblib.load(config_dir)
            consumption = aux['consumption']
            generation = aux['generation']
            public_key = aux['public_key']
            private_key = aux['private_key']

            now = datetime.now()


            if  database.get_sensor_active(generation) == 1 and generation != "None":
                database.update_database(generation)
                database.clean_database_hourly_average(sensor_id=generation, all_sensors=False)
                generation_data = database.get_latest_data_from_sensor(sensor_id=generation)
                generation_timestamp = to_datetime(generation_data[0]).strftime("%Y-%m-%d %H:%M")
                generation_value = generation_data[1]
            else:
                logger.warning(f"⚠️ Recorda seleccionar el sensor de Generació i marcar-lo a l'apartat 'Sensors' per a guardar.")
                generation_timestamp = None
                generation_value = None

            if database.get_sensor_active(consumption) == 1 and consumption != 'None':
                database.update_database(consumption)
                database.clean_database_hourly_average(sensor_id=consumption, all_sensors=False)
                consumption_data = database.get_latest_data_from_sensor(sensor_id=consumption)
                consumption_timestamp = to_datetime(consumption_data[0]).strftime("%Y-%m-%d %H:%M")
                consumption_value = consumption_data[1]
            else:
                logger.warning(f"⚠️ Recorda seleccionar el sensor de Consum i marcar-lo a l'apartat 'Sensors' per a guardar.")
                consumption_timestamp = None
                consumption_value = None


            to_send_string = f"Consumption_{consumption_timestamp}_{consumption_value}_Generation_{generation_timestamp}_{generation_value}_{public_key}_{now}"

            res_certify = blockchain.certify_string(public_key, private_key, to_send_string)

            if res_certify:
                full_path = os.path.join(forecast.models_filepath, "config", "res_certify.pkl")

                if os.path.exists(full_path):
                    data_to_save = joblib.load(full_path)
                else:
                    data_to_save = {}

                now = now.strftime("%Y-%m-%d %H:%M")

                is_success = res_certify['success']
                if is_success:
                    data_to_save[now] = res_certify['response']['transactionHash']
                else:
                    data_to_save[now] = "Error"

                data_to_save = dict(OrderedDict(sorted(data_to_save.items())[-10:]))

                joblib.dump(data_to_save, full_path)

            logger.info("🕒 CERTIFICAT HORARI COMPLETAT")

        else:
            logger.warning(f"Encara no t'has unit a cap comunitat! \n"
                        f"Recorda completar la teva configuració d'usuari des de l'apartat 'configuració' de la pàgina")
    except Exception as e:
        logger.error(f" ❌ [{datetime.now().strftime('%d:%m:%Y %H:%m')}] - ERROR sending hourly task: {e}")

def config_optimized_devices_HA():
    try:
        if (optimalScheduler.consumers == {} and
                optimalScheduler.generators == {} and
                optimalScheduler.energy_storages == {}):
            optimalScheduler.prepare_data_for_optimization()


        today = datetime.today().strftime("%d_%m_%Y")
        full_path = os.path.join(forecast.models_filepath, "optimizations/"+ today +".pkl")
        
        if not os.path.exists(full_path):
            can_optimize = optimize(today=True)
            if can_optimize == "Empty": return

        optimization_db = joblib.load(full_path)

        current_hour = datetime.now().hour

        collections = [
            optimalScheduler.consumers.values(),
            optimalScheduler.generators.values(),
            optimalScheduler.energy_storages.values()
        ]
        for collection in collections:
            for item in collection:
                value, sensor_id, sensor_type = item.controla(config = optimization_db['devices_config'][item.name], current_hour = current_hour)
                database.set_sensor_value_HA(sensor_type, sensor_id, value)
    except Exception as e:
        logger.error(f"❌ [{datetime.now().strftime('%d:%m:%Y %H:%m')}] -  Error configurant horariament un dispositiu a H.A {e}")

schedule.every().day.at("23:30").do(daily_task)
schedule.every().day.at("23:45").do(daily_forecast_task)
schedule.every().day.at("02:00").do(monthly_task)
schedule.every().hour.at(":00").do(certificate_hourly_task)

def run_scheduled_tasks():
    logger.debug("🗓️ SCHEDULER STARTED")
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            # Això evita que el thread mori si una tasca falla
            logger.error(f"❌ Error en l'execució d'una tasca: {e}", exc_info=True)
        time.sleep(1)

scheduler_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
scheduler_thread.start()

#endregion DAILY TASKS


#region DEBUG REGION
@app.route('/panik_function')
def panik_function():

    config_optimized_devices_HA()
    # database.set_sensor_value_HA("number", "number.sonnenbatterie_79259_number_charge", 300)
    # database.set_sensor_value_HA("number", "number.sonnenbatterie_79259_number_discharge", 300)

#endregion DEBUG REGION


# Funció main que encén el servidor web.
def main():
    run(app=app, host=HOSTNAME, port=PORT, quiet=True)


# Executem la funció main
if __name__ == "__main__":
    logger.info("🌳 ExitOS Iniciat")
    main()
