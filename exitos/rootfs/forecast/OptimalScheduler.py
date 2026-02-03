import copy
import requests
import os
import glob
import json
import logging

import sqlDB as db
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution, Bounds


from abstraction.AbsEnergyStorage import AbsEnergyStorage
from abstraction.DeviceRegistry import DEVICE_REGISTRY, create_device_from_config


logger = logging.getLogger("exitOS")

class OptimalScheduler:
    def __init__(self, database: db):

        self.database = database
        self.latitude, self.longitude = database.get_lat_long()

        if self.database.running_in_ha: self.base_filepath = "share/exitos/"
        else: self.base_filepath = "./share/exitos/"

        self.varbound = None
        self.horizon = 24
        self.horizon_min = 1  # 1 = 60 minuts  | 2 = 30 minuts | 4 = 15 minuts
        self.maxiter = 150
        self.timestamps = []

        self.global_consumer_id = ""
        self.global_generator_id = ""
        self.global_consumer_forecast = {
            "forecast_data": [0] * (self.horizon * self.horizon_min),
            "forecast_timestamps": [0] * (self.horizon * self.horizon_min)
        }
        self.global_generator_forecast = {
            "forecast_data": [0] * (self.horizon * self.horizon_min),
            "forecast_timestamps": [0] * (self.horizon * self.horizon_min)
        }

        self.consumers = {}
        self.generators = {}
        self.energy_storages = {}

        self.best_result = 9999
        self.current_result = 0
        self.best_result_balance = None
        self.current_result_balance = None

        self.electricity_prices = []


    def start_optimization(self, consumer_id, generator_id, horizon, horizon_min):
        try:
            self.horizon = horizon
            self.horizon_min = horizon_min

            has_data = self.prepare_data_for_optimization()

            if has_data:
                self.global_consumer_id = consumer_id
                self.global_generator_id = generator_id

                self.global_consumer_forecast['forecast_data'], self.global_consumer_forecast['forecast_timestamps'] = self.get_sensor_forecast_data(consumer_id)
                self.global_generator_forecast['forecast_data'], self.global_generator_forecast['forecast_timestamps'] = self.get_sensor_forecast_data(generator_id)

                self.varbound = self.configure_varbounds()

                self.electricity_prices = self.get_hourly_electric_prices()

                result, cost = self.__optimize()
                total_balance = self.__calc_total_balance(config = result, total = False)
                all_devices_config = self.get_hourly_config_for_device(result)
            else:
                result = None
                cost = []
                total_balance = []
                all_devices_config = []

            return has_data, all_devices_config, cost, total_balance

        except Exception as e:
            logger.error(f"❌ No s'ha pogut realitzar l'optimització: {e}")
            return False, None, None, None


    def prepare_data_for_optimization(self):
        """

        :return:
        """

        configs_saved = [os.path.basename(f) for f in glob.glob(self.base_filepath + "optimizations/configs/*.json")]
        if len(configs_saved) == 0:
            return False

        for config in configs_saved:
            config_path = os.path.join(self.base_filepath, "optimizations/configs", f"{config}")
            with open(config_path, "r",encoding="utf-8") as f:
                current_config = json.load(f)


            dev = create_device_from_config(current_config, self.database)

            device_category = current_config['device_category']
            device_type = current_config['device_type']

            if device_category == "Generator":
                self.generators[device_type] = dev
            elif device_category == "Consumer":
                self.consumers[device_type] = dev
            elif device_category == "EnergyStorage":
                self.energy_storages[device_type] = dev
            else:
                raise ValueError(f"Categoria '{device_category}' desconeguda per al dispositiu {device_type}")

        return True

    def get_sensor_forecast_data(self,sensor_id):
        """
        Obté l'últim forecast del sensor i prepara les dades per a l'optimització

        :param sensor_id: id del sensor a preparar
        :param horizon: hores a simular
        :param horizon_min: minuts per hora 1 = 60 minuts  | 2 = 30 minuts | 4 = 15 minuts
        :return:
        """

        sensor_forecast = self.database.get_data_from_forecast_from_date_and_sensorID(sensor_id=sensor_id, date= datetime.today().strftime('%d-%m-%Y'))

        today = datetime.now()
        start_date = datetime(today.year, today.month, today.day, 0,0)
        end_date = start_date + timedelta(hours = self.horizon - 1)
        self.timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
        hours = []
        for i in range(len(self.timestamps)): hours.append(self.timestamps[i].strftime("%Y-%m-%d %H:%M"))

        sensor_data = []

        for hour in hours:
            if hour in sensor_forecast['timestamp'].values:
                row = sensor_forecast[sensor_forecast['timestamp'] == hour]
                sensor_data.append(row['value'].values[0])
            else:
                sensor_data.append(0)

        return sensor_data, self.timestamps

    def configure_varbounds(self):
        """
        Configura varbounds
        :return:
        """

        lb = []
        ub = []
        index = 0
        num_steps = self.horizon * self.horizon_min


        collections = [
            self.consumers.values(),
            self.generators.values(),
            self.energy_storages.values()
        ]

        # CONSUMERS
        for collection in collections:
            for item in collection:
                item.vbound_start = index

                lb.extend([item.min] * num_steps)
                ub.extend([item.max] * num_steps)
                index += num_steps

                item.vbound_end = index - 1

        bounds = Bounds(lb, ub, True)
        return bounds

    def __optimize(self):
        logger.info(f"🦖 - Començant optimització  a les {datetime.now().strftime('%Y-%m-%d %H:00')}")

        result = differential_evolution(func = self.cost_DE,
                                        popsize = 150,
                                        bounds = self.varbound,
                                        integrality = [1] * len(self.varbound.lb),
                                        maxiter = self.maxiter,
                                        mutation = (0.15, 0.25),
                                        recombination = 0.7,
                                        tol = 0.0001,
                                        strategy = 'best1bin',
                                        init = 'halton',
                                        disp = True,
                                        updating = 'deferred',
                                        callback = self.__update_DE_step,
                                        workers = 1
                                        )


        # if not self.database.running_in_ha:
        logger.warning(" OPTIMIZE FINALITZAT")
        logger.debug(f"     ▫️ Status: {result['message']}")
        logger.debug(f"     ▫️ Total evaluations: {result['nfev']}")
        logger.debug(f"     ▫️ Solution: {result['x']}")
        logger.debug(f"     ▫️ Cost: {result['fun']}")


        return result['x'].copy(), result['fun']

    def cost_DE(self, config):
        return self.__calc_total_balance(config)

    def __update_DE_step(self,bounds, convergence):
        logger.info(f"◽ New Step")
        logger.debug(f"      ▫️ Convergence {convergence}")
        logger.debug(f"      ▫️ Bounds {bounds}")

        logger.debug(f"      ▫️ Current price {self.current_result}")
        logger.debug(f"      ▫️ Best price {self.best_result}")

        if self.current_result < self.best_result:
            logger.debug(f"      ▫️ Updated Best result: {self.best_result} -> {self.current_result}")
            self.best_result = self.current_result
            self.best_result_balance = self.current_result_balance

    def __calc_total_balance(self,config, total = True):

        total_balance = [0] * (self.horizon * self.horizon_min)

        total_consumers = self.__calc_total_balance_consumer(config)
        total_generators = self.__calc_total_balance_generator(config)

        for hour in range(self.horizon * self.horizon_min):
            total_consumers[hour] += self.global_consumer_forecast['forecast_data'][hour]
            total_generators[hour] += self.global_generator_forecast['forecast_data'][hour]

            total_balance[hour] = total_consumers[hour] - total_generators[hour]

        balance_result = self.__calc_total_balance_energy(config, total_balance)

        if not total: return balance_result


        # ajuntem el consum horari en una sola variable global.
        total_price = 0

        for hour in range(self.horizon * self.horizon_min):
            total_price += balance_result[hour] * (self.electricity_prices[hour]/1000)

        self.current_result_balance = balance_result
        self.current_result = total_price

        return total_price

    def __calc_total_balance_consumer(self, config):
        total_consumption = [0] * (self.horizon * self.horizon_min)
        for consumer in self.consumers.values():
            start = consumer.vbound_start
            end = consumer.vbound_end

            res_dict = consumer.simula(config[start:end+1].copy(), self.horizon, self.horizon_min)
            for hour in range(len(res_dict['consumption_profile'])):
                total_consumption[hour] += res_dict['consumption_profile'][hour]

        return total_consumption

    def __calc_total_balance_generator(self, config):
        return [0] * (self.horizon * self.horizon_min)

    def __calc_total_balance_energy(self, config, total_balance):

        total_energy_sources = list(total_balance)
        for energy_storage in self.energy_storages.values():
            start = energy_storage.vbound_start
            end = energy_storage.vbound_end

            res_dict = energy_storage.simula(config[start:end+1], self.horizon, self.horizon_min)

            for hour in range(0, len(total_balance)):
                total_energy_sources[hour] += res_dict['consumption_profile'][hour]

        return total_energy_sources

    def get_hourly_electric_prices(self,):
        today = datetime.today().strftime('%Y%m%d')
        url = f"https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{today}.1"
        response = requests.get(url)

        if response.status_code != 200:
            logger.error(f"❌ No s'ha pogut obtenir el preu de la llum de OMIE: {response.status_code}.")
            return -1

        with open('omie_price_pred.csv', "wb") as file:
            file.write(response.content)

        hourly_prices = []
        with open('omie_price_pred.csv', "r") as file:
            for line in file.readlines()[1:-1]:
                components = line.strip().split(';')
                components.pop(-1)
                hourly_price = float(components[-1])
                hourly_prices.append(hourly_price)
        os.remove('omie_price_pred.csv')

        return_prices = []
        aux = 0
        for i in range(self.horizon):
            for j in range(self.horizon_min):
                return_prices.append(hourly_prices[aux])
            aux += 1
            if aux == 24: aux = 0

        return return_prices

    def get_hourly_config_for_device(self, config):

        collections = [
            self.consumers.values(),
            self.generators.values(),
            self.energy_storages.values(),
        ]
        all_devices_config = {}

        for collection in collections:
            for item in collection:
                start = item.vbound_start
                end = item.vbound_end

                device_config = config[start:end+1]

                all_devices_config[item.name] = device_config

        return all_devices_config