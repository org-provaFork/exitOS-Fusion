from datetime import timedelta
import time

import joblib
import numpy as np
import pandas as pd
import holidays
import logging
import os
import glob
import requests
import json
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import mean_absolute_error

from forecast.ForecastMetrics import ForecastMetrics

logger = logging.getLogger("exitOS")


class Forecaster:
    def __init__(self, debug=False):
        """
        Constructor per defecte del Forecaster
        """

        self.debug = debug
        self.search_space_config_file = 'resources/search_space.conf'
        self.db = dict()
        self.metrics = ForecastMetrics(debug=debug)

        if "HASSIO_TOKEN" in os.environ:
            self.models_filepath = "share/exitos/"
        else:
            self.models_filepath = "./share/exitos/"

    @staticmethod
    def windowing_group(dataset, look_back_start=24, look_back_end=48):
        """
        Funció per crear les variables del windowing. \n
        Treballa sobre un dataset i inclou la variable objectiu. \n
        Les variables creades es diràn com les originals (legacy) i s'afagirà '_' amb el número de desplaçament. \n
        Es tindràn en compte les hores en el rang [ini,fi)

        :param dataset: Dataframe amb datetime com a índex
        :type dataset: pd.DataFrame
        :param look_back_start: On comença la finestra ( 24 -> el dia anterior si és horari)
        :param look_back_end: On acaba el número d'observacions (48 -> el dia anterior si és horari)
        :return: Dataset amb les cariables desplaçades en columnes noves
        :rtype: pd.DataFrame
        """

        ds = dataset.copy()
        shifted_columns = {}

        for col in ds.columns:
            if col != 'timestamp':
                for j in range(look_back_start, look_back_end):
                    shifted_columns[f"{col}_{j}"] = ds[col].shift(j)

        ds = pd.concat([ds, pd.DataFrame(shifted_columns, index=ds.index)], axis=1)

        return ds

    @staticmethod
    def windowing_univariant(dataset, look_back_start=24, look_back_end=48, variable=''):
        """
        Funció per crear les variables del windowing. \n
        Treballa sobre un dataset i inclou la variable objectiu. \n
        Les variables creades es diràn com les originals (legacy) i s'afagirà '_' amb el número de desplaçament. \n
        Es tindràn en compte les hores en el rang [ini,fi)

        :param dataset: Dataframe amb datetime com a índex
        :type dataset: pd.DataFrame
        :param look_back_start: On comença la finestra ( 24 -> el dia anterior si és horari)
        :param look_back_end: On acaba el número d'observacions (48 -> el dia anterior si és horari)
        :param variable: Variable a transformar en variables del windowing.
        :return: Dataset amb les cariables desplaçades en columnes noves
        :rtype: pd.DataFrame
        """

        ds = dataset.copy()
        for i in range(0, len(ds.columns)):
            if ds.columns[i] == variable:
                for j in range(look_back_start, look_back_end - 1):
                    ds[ds.columns[i] + '_' + str(j)] = ds[ds.columns[i]].shift(j)

        return ds

    def do_windowing(self, data, look_back={-1: [25, 48]}):
        """
        Aplica el Windowing en consequencia al look_back indicat.\n
        - None -> no aplica el windowing \n
        - Diccionari on la clau és la variable a fer windowing i el valor la finestra que s'ha d'aplicar \n
        - Les claus son Strings, indicant el nom de la columna a aplicar windowing
        - Si com a clau es dona -1, la finestra aplicara a totes les variables NO especificades individualment.
        - Els valors són els que defineixen la finestra a aplicar, i poden ser:
            - [ini, fi]
            - [ini, fi, salt]
        :param data: dataframe de dades
        :param look_back: Windowing a aplicar
        :return: dataframe de dades preparades per el model de forecasting.
        """
        if look_back is not None:
            # windowing  de tots els grups (NO individuals)
            if -1 in look_back.keys():  # si l'indicador és -1 volem un grup
                aux = look_back[-1]  # recuperem els valors de la finestra pel grup

                # recuperem les que es faran soles
                keys = list()
                for i in look_back.keys():
                    if i != -1:
                        keys.append(i)

                dad = data.copy()  # copiem el dataset per no perdre les que aniran soles
                dad = dad.drop(columns=keys)  # eliminem les que van soles

                # windowing de tot el grup
                dad = self.windowing_group(dad, aux[0], aux[1])

                # afegim les que haviem tret abans
                for i in keys:
                    dad[i] = data[i]

            else:
                # cas de que no tinguem grups, son tots individuals
                dad = data.copy()  # copiem el dataset
                keys = list()
                for i in look_back.keys():
                    if i != -1:
                        keys.append(i)

            # windowing de la resta que es fan 1 a 1
            variables = [col for col in data.columns if col not in keys]
            for i in variables:
                if i.startswith('timestamp'): continue
                aux = look_back[-1]
                dad = self.windowing_univariant(dad, aux[0], aux[1], i)
        else:
            # cas de no tenir windowing
            dad = data.copy()

        return dad

    @staticmethod
    def timestamp_to_attrs(dad, extra_vars):
        """
        Afageix columnes derivades de l'índex temporal al DataFreame 'dad' segons les opcions indicades en 'extra_vars'. \n'
        :param dad: Dataframe amb un índex timestamp
        :type dad: pd.DataFrame
        :param extra_vars: Diccionari amb opcions per a generar columnes adicionals ('variables', 'festius')
        :type extra_vars: dict
        :return: El mateix DataFrame amb les noves columnes afegides.
        """

        if 'timestamp' in dad.columns:
            dad.index = pd.to_datetime(dad['timestamp'])

        if not extra_vars:
            # si extra_vars és None o buit, no cal fer res
            return dad

        # afegim columnes basades en l'índex temporal
        if 'variables' in extra_vars:
            for var in extra_vars['variables']:
                if var == 'Dia':
                    dad['Dia'] = dad.index.dayofweek  # Dia de la setmana (0 = Dll, 6 = Dg)
                elif var == 'Hora':
                    dad['Hora'] = dad.index.hour  # Hora del dia
                elif var == 'Mes':
                    dad['Mes'] = dad.index.month  # Mes de l'any
                elif var == 'Minut':
                    dad['Minut'] = dad.index.minute  # Minut de l'hora

        # Afegim columnes per a dies festius
        if 'festius' in extra_vars:
            festius = extra_vars['festius']

            # Necessitem passar els anys explícitament per a que holidays generi les dades
            years = dad.index.year.unique()

            if len(festius) == 1:
                # Festius d'un sol país
                h = holidays.country_holidays(festius[0], years=years)
            elif len(festius) == 2:
                # festius d'un sol país amb una regió específica
                h = holidays.country_holidays(festius[0], festius[1], years=years)
            else:
                raise ValueError("La clau 'festius' només suporta 1 o 2 paràmetres (país i opcionalment regió)")

            # Afageix una columna booleana indicant si cada dia es festiu
            # Convertim les dates festives a string per comparar amb strftime (evita error Str vs Date)
            h_str = {d.strftime('%Y-%m-%d') for d in h.keys()}
            dad['festius'] = dad.index.strftime('%Y-%m-%d').isin(h_str)

        if 'timestamp' in dad.columns:
            dad.drop(columns=['timestamp'], inplace=True)

        return dad

    @staticmethod
    def colinearity_remove(data, y, level=0.9):
        """
        Elimina les colinearitats entre les variables segons el nivell indicat
        :param data: Dataframe amb datetime com a índex
        :type data: pd.DataFrame
        :param y: Variable objectiu (per mirar que no la eliminem!)
        :param level: el percentatge de correlació de pearson per eliminar variables. None per no fer res
        :return:
            - dataset - Dataset amb les variables eliminades
            - to_drop - Les variables que ha eliminat
        """

        if level is None:
            dataset = data
            to_drop = None
        else:
            # eliminem els atributs massa correlacionats
            corr_matrix = data.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper.columns if any(upper[column] > level)]
            if np.array(to_drop == y).any():
                del to_drop[to_drop == y]
            data.drop(to_drop, axis=1, inplace=True)
            dataset = data.copy()

        return [dataset, to_drop]

    @staticmethod
    def scalate_data(data, input_scaler=None):
        """
        Escala les dades del dataset
        :param data: Dataset a escalar
        :param scaler: Mètode a usar per escalar
        :return: [Dataset , scaler]
        """
        dad = data.copy()
        scaler = None

        if input_scaler is not None:
            if input_scaler == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                dad = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
            elif input_scaler == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                dad = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
            elif input_scaler == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                dad = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
            else:
                raise ValueError('Atribut Scaler no definit')
        else:
            scaler = None

        return dad, scaler

    @staticmethod
    def get_attribs(X, y, method=None):
        """
        Fa una selecció d'atributs
        :param X: Array amb les dades
        :param y: Array amb les dades
        :param method:
            - None: No fa res
            - integer: selecciona el número de features que s'indiquin
            - PCA: aplica un PCA per comprimir el dataset
        :return:
        """

        if method is None:
            model_select = []
            X_new = X
        elif method == 'Tree':
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.feature_selection import SelectFromModel

            clf = ExtraTreesRegressor(n_estimators=50)
            clf = clf.fit(X, y)
            model_select = SelectFromModel(clf, prefit=True)
            X_new = model_select.transform(X)
        elif type(method) == int:
            from sklearn.feature_selection import SelectKBest, f_classif
            model_select = SelectKBest(f_classif, k=method)
            X_new = model_select.fit_transform(X, y)
        else:
            raise ValueError('Atribut de mètode de selecció no definit')

        return [model_select, X_new, y]

    def Model(self, X, y, algorithm=None, params=None, max_time=None):
        """
        Realitza un randomized search per trovar una bona configuració de l'algorisme indicat, o directament, es crea amb els paràmetres indicats.
        :param X:
        :param y:
        :param algorithm:
        :param params:
        :param max_time:
        :return:
        """

        import json

        with open(self.search_space_config_file) as json_file:
            d = json.load(json_file)

        if type(params) == dict:
            try:
                impo1 = d[algorithm][2]
                impo2 = d[algorithm][3]
            except:
                raise ValueError("Undefined Firecasting Algorithm!")

            a = __import__(impo1, globals(), locals(), [impo2])
            forecast_algorithm = eval("a. " + impo2)

            f = forecast_algorithm()
            f.set_params(**params)
            f.fit(X, y)
            score = 'none'
            return [f, score]
        elif params is None:  # no tenim paràmetres. Els busquem
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

            from sklearn.model_selection import ParameterSampler
            from sklearn.metrics import mean_absolute_error
            import time

            best_mae = np.inf

            # preparem la llista d'algorismes que volem provar
            if algorithm is None:
                # si no ens passen res els probem tots
                algorithm_list = list(d.keys())
            elif isinstance(algorithm, list):
                # passen una llista a probar
                algorithm_list = algorithm
            else:
                # ens passen només 1
                algorithm_list = [algorithm]

            best_overall_model = None
            best_overall_score = float("inf")
            # per cada algorisme a probar....
            for i in range(len(algorithm_list)):
                random_grid = d[algorithm_list[i]][0]

                if max_time is None:
                    iters = d[algorithm_list[i]][1]
                else:
                    iters = max_time

                impo1 = d[algorithm_list[i]][2]
                impo2 = d[algorithm_list[i]][3]

                if self.debug:
                    logger.info("  ")
                    logger.info(
                        f"Començant a optimitzar:  {algorithm_list[i]}  - Algorisme {str(algorithm_list.index(algorithm_list[i]) + 1)}  de  {str(len(algorithm_list))} - Maxim comput aprox (segons): {str(iters)}")

                sampler = ParameterSampler(random_grid, n_iter=np.iinfo(np.int64).max)

                a = __import__(impo1, globals(), locals(),
                               [impo2])  # equivalent a (import sklearn.svm as a) dinamicament
                forecast_algorithm = eval("a. " + impo2)

                try:
                    # creem i evaluem els models 1 a 1
                    t = time.perf_counter()

                    if self.debug:
                        logger.debug(f"Maxim {str(len(sampler))}  configuracions a probar!")
                        j = 0

                    best_model = None
                    best_mae = float("inf")

                    for params in sampler:
                        try:
                            regr = forecast_algorithm(**params)
                        except Exception as e:
                            logger.warning(f"Failed with params: {params}")
                            continue

                        pred_test = regr.fit(X_train, y_train).predict(X_test)
                        act = mean_absolute_error(y_test, pred_test)
                        if best_mae > act:
                            best_model = regr
                            best_mae = act

                        if self.debug:
                            logger.info("\r")
                            j += 1
                            logger.debug(
                                f"{str(j)} / {str(len(sampler))} Best MAE: {str(best_mae)} ({type(best_model).__name__}) Last MAE: {str(act)}  Elapsed: {str(time.perf_counter() - t)} s ")

                        if (time.perf_counter() - t) > iters:
                            if self.debug:
                                logger.debug("Algorisme " + algorithm_list[
                                    i] + " -- Abortat per Maxim comput aprox (segons): " + str(iters))
                                break

                except Exception as e:
                    logger.warning(f"WARNING: Algorisme {algorithm_list[i]},  -- Abortat per Motiu: {str(e)}")
                    continue

                if best_model is not None:
                    best_model.fit(X, y)
                    if best_mae < best_overall_score:
                        best_overall_model = best_model
                        best_overall_score = best_mae

            return [best_overall_model, best_overall_score]
        else:
            raise ValueError('Paràmetres en format incorrecte!')

    @staticmethod
    def prepare_dataframes(sensor, meteo, extra_sensors):
        """
        Prepara els df juntant-los en un de sol.
        Ara fa un resample horari promig (mean) en lloc de quedar-se amb el primer valor.
        :param sensor: Sensor objectiu del model
        :param meteo: Dades meteorològiques (pot ser None)
        :param extra_sensors: Sensors extra que es volen usar pel model (pot ser empty)
        """
        merged_data = pd.DataFrame()

        # 1. Preparar sensor principal (Target)
        if sensor is not None and not sensor.empty:
            sensor = sensor.copy() # Evitar SettingWithCopy warning
            sensor['timestamp'] = pd.to_datetime(sensor['timestamp']).dt.tz_localize(None)
            sensor.set_index('timestamp', inplace=True)
            
            # Assegurar que 'value' és numèric
            if 'value' in sensor.columns:
                sensor['value'] = pd.to_numeric(sensor['value'], errors='coerce')
            
            # Resample horari fent la mitjana (només columnes numèriques per evitar errors)
            sensor = sensor.resample('h').mean(numeric_only=True)
            merged_data = sensor.copy()
        
        # 2. Preparar dades meteo
        if meteo is not None and not meteo.empty:
            meteo = meteo.copy()
            meteo['timestamp'] = pd.to_datetime(meteo['timestamp']).dt.tz_localize(None)
            meteo.set_index('timestamp', inplace=True)
            # Meteo ja sol venir horària, però per seguretat fem resample
            meteo = meteo.resample('h').mean(numeric_only=True)
            
            if merged_data.empty:
                merged_data = meteo
            else:
                merged_data = pd.merge(merged_data, meteo, left_index=True, right_index=True, how='outer')

        # 3. Preparar sensors extra
        if extra_sensors is not None:
             for key, df_extra in extra_sensors.items():
                if df_extra is not None and not df_extra.empty:
                    df = df_extra.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                    df.set_index('timestamp', inplace=True)
                    
                    # Assegurar que 'value' és numèric
                    if 'value' in df.columns:
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')

                    # Resample horari
                    df = df.resample('h').mean(numeric_only=True)
                    
                    if merged_data.empty:
                        merged_data = df
                    else:
                        merged_data = pd.merge(merged_data, df, left_index=True, right_index=True, how='outer')

        # Si després de tot no tenim dades, retornem el sensor original processat (o buit)
        if merged_data.empty:
            return pd.DataFrame()

        # Rename value_x to value if needed (fix legacy merging naming issues)
        if "value" not in merged_data.columns and "value_x" in merged_data.columns:
             merged_data.rename(columns={'value_x': 'value'}, inplace=True)

        # Reset index per tornar a tenir timestamp com a columna (compatible amb la resta de codi)
        merged_data.reset_index(inplace=True)
        
        return merged_data

    def create_model(self, data, sensors_id, y, lat, lon, algorithm=None, params=None, escalat=None,
                         max_time=None, filename='newModel', meteo_data: pd.DataFrame = None, extra_sensors_df=None,
                         look_back=None):
        """
        Funció per crear, guardar i configurar el model de forecasting.

        :param lon:
        :param lat:
        :param extra_sensors_df:
        :param data: Dataframe amb datetime com a índex
        :param sensors_id:
        :param y: Nom de la columna amb la variable a predir
        :param filename:
        :param max_time:
        :param escalat:
        :param params:
        :param algorithm:
        :param meteo_data: Dades meteorològiques de la data
        :param look_back: Configuració del windowing (per defecte {-1: [25, 48]})

        """
        
        # Reiniciar mètriques per a aquest model
        self.metrics = ForecastMetrics(debug=self.debug)

        extra_vars = {'variables': ['Dia', 'Hora', 'Mes'], 'festius': ['ES', 'CT']}
        feature_selection = 'Tree'
        colinearity_remove_level = 0.9
        if look_back is None:
            # Per defecte mirar les últimes 24h (1..24) enlloc d'ahir (25..48)
            look_back = {-1: [1, 24]}

        # Descarregar dades meteo si no es proporcionen
        if meteo_data is not None and not data.empty:
            start_date = data['timestamp'].min().strftime("%Y-%m-%d")
            end_date = data['timestamp'].max().strftime("%Y-%m-%d")

            logger.info(f"⛅ Descarregant dades meteo històriques de {start_date} a {end_date}")

            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,"
                f"precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,"
                f"cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,"
                f"vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,"
                f"winddirection_100m,windgusts_10m,shortwave_radiation,direct_radiation,"
                f"diffuse_radiation,direct_normal_irradiance,terrestrial_radiation"
            )

            try:
                response = requests.get(url).json()
                hourly = response.get("hourly", {})
                timestamps = pd.to_datetime(hourly["time"])
                meteo_data = pd.DataFrame(hourly)
                meteo_data["timestamp"] = timestamps
                meteo_data.drop(columns=["time"], inplace=True)
            except Exception as e:
                logger.error(f"❌ No s'han pogut descarregar les dades meteo històriques: {e}")
                meteo_data = None

        #PREP PAS 0 - preparar els df de meteo-data i dades extra
        merged_data = self.prepare_dataframes(data, meteo_data, extra_sensors_df)
        
        # Interpolació temporal dels NaNs inicials
        if 'timestamp' in merged_data.columns:
            merged_data.set_index('timestamp', inplace=True)
            merged_data.interpolate(method='time', inplace=True)
            merged_data.reset_index(inplace=True)
            
        merged_data.bfill(inplace=True)
        
        # VALIDACIÓ PAS 0
        self.metrics.validate_dataframe_preparation(data, meteo_data, extra_sensors_df, merged_data)

        if merged_data.empty:
            logger.error(f"\n ************* \n ❌ No hi ha dades per a realitzar el Forecast \n *************")
            return

        # PAS 1 - Fer el Windowing
        dad_before_windowing = merged_data.copy()
        dad = self.do_windowing(merged_data, look_back)
        
        # VALIDACIÓ PAS 1
        self.metrics.validate_windowing(dad_before_windowing, dad, look_back)

        # PAS 2 - Crear variable dia_setmana, hora, més i meteoData
        dad = self.timestamp_to_attrs(dad, extra_vars)
        
        # VALIDACIÓ PAS 2
        self.metrics.validate_temporal_features(dad, extra_vars)

        # PAS 3 - Treure Col·linearitats
        dad_before_colinearity = dad.copy()
        [dad, to_drop] = self.colinearity_remove(dad, y, level=colinearity_remove_level)
        colinearity_remove_level_to_drop = to_drop
        
        # VALIDACIÓ PAS 3
        self.metrics.validate_colinearity_removal(dad_before_colinearity, dad, to_drop, y, colinearity_remove_level)

        # PAS 4 - Gestió de NaN (Interpolació + Eliminació)
        dad.replace([np.inf, -np.inf], np.nan, inplace=True)
        dad_before_nan = dad.copy()
        
        # Interpolació temporal
        X = dad.interpolate(method='time')
        X = X.bfill()
        X = X.dropna()
        
        # VALIDACIÓ PAS 4
        self.metrics.validate_nan_handling(dad_before_nan, X)
        
        # PAS 5 - Desfer el dataset i guardar matrius X i y
        nomy = y
        y = pd.to_numeric(X[nomy], errors='raise')
        del X[nomy]
        
        # Divisió Train/Validation/Test (60/20/20) 

        # PAS 6 - Escalat
        X_before_scaling = X.copy()
        X, scaler = self.scalate_data(X, escalat)
        
        # VALIDACIÓ PAS 6
        self.metrics.validate_scaling(X_before_scaling, X, escalat)

        # PAS 7 - Seleccionar atributs
        [model_select, X_new, y_new] = self.get_attribs(X, y, feature_selection)
        
        # VALIDACIÓ PAS 7
        X_before_selection = X if isinstance(X, np.ndarray) else X.values
        self.metrics.validate_feature_selection(X_before_selection, X_new, feature_selection)

        # PAS 8 - Crear el model
        # Dividir les dades en 60% Train, 20% Validation, 20% Test
        train_idx = int(len(X_new) * 0.6)
        val_idx = int(len(X_new) * 0.8) # 60% + 20% = 80%

        # Gestió segons tipus de dades (pandas o numpy)
        if isinstance(X_new, pd.DataFrame):
            X_train = X_new.iloc[:train_idx]
            X_val = X_new.iloc[train_idx:val_idx]
            X_test = X_new.iloc[val_idx:]
        else:
            X_train = X_new[:train_idx]
            X_val = X_new[train_idx:val_idx]
            X_test = X_new[val_idx:]
            
        y_train = y_new[:train_idx]
        y_val = y_new[train_idx:val_idx]
        y_test = y_new[val_idx:]

        training_start = time.time()
        # Entrenem amb X_train (60% només)
        [model, score] = self.Model(X_train, y_train.values, algorithm, params, max_time=max_time)
        training_time = time.time() - training_start
        
        # VALIDACIÓ INTERMÈDIA (20% Validation)
        # Validem la bondat de l'entrenament sobre el conjunt de validació
        try:
            val_score = model.score(X_val, y_val.values)
            logger.info(f"📊 Validation Score: {val_score:.4f}")
        except Exception as e:
            logger.warning(f"⚠️ No s'ha pogut calcular el Validation Score: {e}")

        # VALIDACIÓ FINAL (20% Test - Dades ocultes finals)
        y_pred = model.predict(X_test)
        
        # Passem X_test i y_test per calcular les mètriques reals sobre dades noves
        self.metrics.validate_model_training(X_test, y_test.values, y_pred, algorithm, score, training_time)
        
        # Comparar amb baselines (usant dades de test)
        last_val_value = y_val.iloc[-1] if hasattr(y_val, 'iloc') else y_val[-1]
        self.metrics.compare_with_baseline(y_test.values, y_pred, last_history_value=last_val_value)

        # PAS 9 - Guardar el model i les mètriques
        if algorithm is None:
            self.db['max_time'] = max_time
            self.db['algorithm'] = "AUTO"
        else:
            self.db['params'] = params
            self.db['algorithm'] = algorithm

        if meteo_data is not None:
            self.db['meteo_data'] = meteo_data
            self.db['meteo_data_is_selected'] = True
        else:
            self.db['meteo_data'] = None
            self.db['meteo_data_is_selected'] = False

        self.db['model'] = model
        self.db['scaler'] = scaler
        self.db['scaler_name'] = escalat
        self.db['model_select'] = model_select
        self.db['colinearity_remove_level_to_drop'] = colinearity_remove_level_to_drop
        self.db['extra_vars'] = extra_vars
        self.db['look_back'] = look_back
        self.db['score'] = score
        self.db['objective'] = nomy
        self.db['initial_data'] = data
        self.db['sensors_id'] = sensors_id
        self.db['extra_sensors'] = extra_sensors_df
        self.db['lat'] = lat
        self.db['lon'] = lon
        
        # Guardar mètriques
        self.db['metrics'] = self.metrics.get_summary()
        self.db['train_val_test_split'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }

        self.save_model(filename)

        if self.debug:
            logger.debug('Model guardat! Score: ' + str(score))

    def forecast(self, data, y, model, future_steps=48):
        """
        Realitza prediccions de forma RECURSIVA per permetre l'ús de lags recents (1..24h).
        """

        # PAS 1 - Obtenir els valors del model
        model_select = self.db.get('model_select', [])
        scaler = self.db['scaler']
        colinearity_remove_level_to_drop = self.db.get('colinearity_remove_level_to_drop', [])
        extra_vars = self.db.get('extra_vars', {})
        look_back = self.db.get('look_back', {-1: [1, 24]})
        
        # Preparem l'històric inicial
        history_df = data.copy()
        if 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.tz_localize(None)
            history_df.set_index('timestamp', inplace=True)
            
        # Interpolació temporal inicial per assegurar continuïtat
        history_df.interpolate(method='time', inplace=True)
        history_df.bfill(inplace=True) # Seguretat extra
        
        predictions = []
        future_indexes = []
        
        # Bucle de predicció recursiva
        for i in range(future_steps):
            last_timestamp = history_df.index[-1]
            next_timestamp = last_timestamp + pd.Timedelta(hours=1)
            future_indexes.append(next_timestamp)
            
            # Creem una nova fila (buid) per al següent pas
            new_row = pd.DataFrame(index=[next_timestamp], columns=history_df.columns)
            
            # Omplim amb persistència les variables exògenes
            for col in new_row.columns:
                if col != y: 
                     new_row[col] = history_df[col].iloc[-1]
                else:
                     new_row[col] = np.nan
                     
            # Afegim la nova fila a l'històric de treball (només la cua necessària per velocitat)
            working_window = pd.concat([history_df.tail(200), new_row])
            
            # --- PIPELINE DE PREPARACIÓ ---
            
            # 1. Windowing
            df_windowed = self.do_windowing(working_window, look_back)
            
            # Ens quedem només amb l'última fila
            current_step_df = df_windowed.tail(1).copy()
            
            # 2. Atributs temporals
            current_step_df = self.timestamp_to_attrs(current_step_df, extra_vars)
            
            # 3. Eliminar colinearitats
            if colinearity_remove_level_to_drop:
                existing_cols = [col for col in colinearity_remove_level_to_drop if col in current_step_df.columns]
                current_step_df.drop(existing_cols, axis=1, inplace=True)
                
            # 4. Eliminar variable objectiu 'y'
            if y in current_step_df.columns:
                del current_step_df[y]
                
            # 5. Gestió de NaNs
            current_step_df.bfill(inplace=True)
            current_step_df.fillna(0, inplace=True)
            
            # 6. Escalat
            if scaler:
                try:
                    current_step_df = pd.DataFrame(scaler.transform(current_step_df), index=current_step_df.index, columns=current_step_df.columns)
                except Exception:
                    pass

            # 7. Selecció d'atributs
            if model_select:
                try:
                     current_step_array = model_select.transform(current_step_df)
                except:
                     current_step_array = current_step_df.values
            else:
                current_step_array = current_step_df.values

            # --- PREDICCIÓ ---
            try:
                pred_val = float(model.predict(current_step_array)[0])
            except Exception as e:
                logger.error(f"Error en predicció pas {i}: {e}")
                pred_val = 0.0

            predictions.append(pred_val)
            
            # Actualitzem el valor predit a l'històric per a la següent iteració
            new_row[y] = pred_val
            history_df = pd.concat([history_df, new_row])
            
        forecast_output = pd.DataFrame(
            predictions,
            index=future_indexes,
            columns=[y]
        )
        
        # Recalculem el passat en mode batch per al gràfic 'fit'
        df_fit = self.do_windowing(data, look_back)
        df_fit = self.timestamp_to_attrs(df_fit, extra_vars)
        if colinearity_remove_level_to_drop:
             existing_cols = [col for col in colinearity_remove_level_to_drop if col in df_fit.columns]
             df_fit.drop(existing_cols, axis=1, inplace=True)
        if y in df_fit.columns:
            real_values_column = df_fit[y]
            del df_fit[y]
        else:
            real_values_column = pd.Series()
            
        df_fit.bfill(inplace=True)
        df_fit.fillna(0, inplace=True)
        
        if scaler:
             df_fit = pd.DataFrame(scaler.transform(df_fit), index=df_fit.index, columns=df_fit.columns)
        if model_select:
             df_fit = model_select.transform(df_fit)
             
        out = pd.DataFrame(model.predict(df_fit), index=real_values_column.index, columns=[y])
        
        final_prediction = pd.concat([out, forecast_output])
        
        return final_prediction, real_values_column,  self.db['sensors_id']

    def save_model(self, model_filename):
        """
        Guarda el model en un arxiu .pkl i l'elimina de la base de daades interna del forecast (self.db)
        :param model_filename: Nom que es vol donar al fitxer, si és nul serà "savedModel"
        """
        full_path = os.path.join(self.models_filepath,'forecastings/', model_filename + ".pkl")
        os.makedirs(self.models_filepath + 'forecastings', exist_ok=True)

        if os.path.exists(full_path):
            logger.warning(f"El fitxer {full_path} ja existeix. S'eliminarà.")
            os.remove(full_path)

        joblib.dump(self.db, full_path)
        logger.info(f"  💾 Model guardat al fitxer {model_filename}.pkl")

        self.db.clear()

    def load_model(self, model_filename):
        self.db = joblib.load(self.models_filepath + 'forecastings/' + model_filename)
        logger.info(f"  💾 Model carregat del fitxer {model_filename}")
