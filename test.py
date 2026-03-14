import unittest
import pandas as pd
from scheduler_three_market import get_daa_schedule, get_ida_schedule, get_idc_schedule

"""
Unit tests for the three-market scheduling functions can be run with the following command:
python -m unittest test.py
or
py -3.xx -m unittest test.py
on windows systems.
"""


class FunctionTests(unittest.TestCase):
    def test_increasing_price_vectors(self):
        # Test data
        pv_vector = [1] * 16 + [0] * (96 - 16)
        daa_price_vector = [0] * 24 + [1] * (16) + [0] * (96 - 40)
        ida_price_vector = [0] * 48 + [2] * (16) + [0] * (96 - 48 - 16)
        idc_price_vector = [0] * 72 + [3] * (16) + [0] * (96 - 72 - 16)

        p_limit = 1.0  # kW
        storage_capacity = 4.0  # kWh
        p_charge_max = 1.0  # kW
        p_discharge_max = 1.0  # kW
        number_of_cycles = 1.0  # cycles per day
        efficiency = 1.0  # round-trip efficiency
        start_soc = 0.0  # 0% initial state of charge
        end_soc = 0.0  # 0% final state of charge

        # Get schedules
        pv_output, daa_price, p_charge_daa, p_discharge_daa, soc_daa, p_curtailed_daa, injection_power_daa = get_daa_schedule(pv_vector,
                                                                                                  daa_price_vector, 
                                                                                                  p_limit, 
                                                                                                  storage_capacity, 
                                                                                                  p_charge_max, 
                                                                                                  p_discharge_max, 
                                                                                                  number_of_cycles, 
                                                                                                  efficiency, 
                                                                                                  start_soc, 
                                                                                                  end_soc)

        ida_price, p_charge_ida, p_discharge_ida, p_close_charge_daa, p_close_discharge_daa, p_curtailed_ida, \
        p_close_curtailed_daa, p_curtailed_daa_ida ,p_charge_daa_ida, p_discharge_daa_ida, soc_ida, injection_power_ida = get_ida_schedule(
                                                                                                                        ida_price_vector, 
                                                                                                                        pv_output,
                                                                                                                        p_limit, 
                                                                                                                        storage_capacity, 
                                                                                                                        p_charge_max, 
                                                                                                                        p_discharge_max,
                                                                                                                        p_charge_daa,
                                                                                                                        p_discharge_daa,
                                                                                                                        p_curtailed_daa,
                                                                                                                        number_of_cycles, 
                                                                                                                        efficiency, 
                                                                                                                        start_soc, 
                                                                                                                        end_soc)
    
        idc_price, p_charge_idc, p_discharge_idc, p_close_charge_daa_ida, p_close_discharge_daa_ida, p_curtailed_idc, \
        p_close_curtailed_daa_ida, p_curtailed_daa_ida_idc, p_charge_daa_ida_idc, p_discharge_daa_ida_idc, soc_idc, injection_power_idc = get_idc_schedule(    
                                                                                                                                idc_price_vector,
                                                                                                                                pv_output, 
                                                                                                                                p_limit, 
                                                                                                                                storage_capacity, 
                                                                                                                                p_charge_max, 
                                                                                                                                p_discharge_max,
                                                                                                                                p_charge_daa_ida,
                                                                                                                                p_discharge_daa_ida,
                                                                                                                                p_curtailed_daa_ida,
                                                                                                                                number_of_cycles, 
                                                                                                                                efficiency, 
                                                                                                                                start_soc, 
                                                                                                                                end_soc
                                                                                                                            )
        # Compile results into a DataFrame
        results = pd.DataFrame({
                            # prices and PV output
                           'daa_price': daa_price,
                           'ida_price': ida_price,
                           'idc_price': idc_price,
                           'pv_output': pv_output,
                            # DAA results
                           'p_charge_daa': p_charge_daa,
                           'p_discharge_daa': p_discharge_daa,
                           'p_curtailed_daa': p_curtailed_daa,
                           'soc_daa': soc_daa,
                           'injection_power_daa': injection_power_daa,
                           # IDA results
                           'p_charge_ida': p_charge_ida,
                           'p_discharge_ida': p_discharge_ida,
                           'p_close_charge_daa': p_close_charge_daa,
                           'p_close_discharge_daa': p_close_discharge_daa,
                           'p_curtailed_ida': p_curtailed_ida,
                           'p_close_curtailed_daa': p_close_curtailed_daa,
                           'p_charge_daa_ida': p_charge_daa_ida,
                           'p_discharge_daa_ida': p_discharge_daa_ida,
                           'p_curtailed_daa_ida': p_curtailed_daa_ida,
                           'soc_ida': soc_ida,
                           'injection_power_ida': injection_power_ida,
                           # IDC results
                           'p_charge_idc': p_charge_idc,
                           'p_discharge_idc': p_discharge_idc,
                           'p_close_charge_daa_ida': p_close_charge_daa_ida,
                           'p_close_discharge_daa_ida': p_close_discharge_daa_ida,
                           'p_curtailed_idc': p_curtailed_idc,
                           'p_close_curtailed_daa_ida': p_close_curtailed_daa_ida,
                           'p_charge_daa_ida_idc': p_charge_daa_ida_idc,
                           'p_discharge_daa_ida_idc': p_discharge_daa_ida_idc,
                           'p_curtailed_daa_ida_idc': p_curtailed_daa_ida_idc,
                           'soc_idc': soc_idc,
                           'injection_power_idc': injection_power_idc
                          }
                         )

        ##### Calculate profits
        # Calculate annual profit from PV Output multiplied with DAA prices as reference
        results['pv_direct_market_profit_reference'] = results.daa_price * results.pv_output * 15 / 60
        # Only sum up profit at positive market prices, because plant will be curtailed at other times
        condition_daa_positive = results.daa_price >= 0
        # annual_pv_production_excl_negative_market_price_hours = results.loc[condition_daa_positive, 'pv_output'].sum() * 15 / 60
        annual_pv_direct_market_profit_reference = results.loc[condition_daa_positive, 'pv_direct_market_profit_reference'].sum()

        # Calculate annual profit on DAA market
        results['injected_power_direct_market_profit_daa'] = results.daa_price * results.injection_power_daa * 15 / 60
        annual_injected_energy_direct_market_profit_daa = results.injected_power_direct_market_profit_daa.sum()

        # Calculate annual profit on IDA market
        results['ida_market_profit'] = (results['p_discharge_ida'] - results['p_charge_ida'] 
                                        + results['p_close_charge_daa'] - results['p_close_discharge_daa'] 
                                        + results['p_close_curtailed_daa'] - results['p_curtailed_ida']) * results['ida_price']  / 4
        annual_ida_market_profit = results['ida_market_profit'].sum()

        # Calculate annual profit on IDC market
        results['idc_market_profit'] = (results['p_discharge_idc'] - results['p_charge_idc'] 
                                        + results['p_close_charge_daa_ida'] - results['p_close_discharge_daa_ida'] 
                                        + results['p_close_curtailed_daa_ida'] - results['p_curtailed_idc']) * results['idc_price']  / 4
        annual_idc_market_profit = results['idc_market_profit'].sum()
        
        # Assertions
        self.assertEqual(annual_injected_energy_direct_market_profit_daa, 4.0)
        self.assertEqual(annual_ida_market_profit, 8.0)
        self.assertEqual(annual_idc_market_profit, 12.0)

    def test_curtailment_1(self):
        # Test data
        pv_vector = [1] * 20 + [0] * (96 - 20)
        daa_price_vector = [-1] * 24 + [1] * 16 + [-1] * (96 - 40)
        ida_price_vector = [2] * 4 + [-1] * 20 + [2] * 16 + [-1] * (96 - 40)
        idc_price_vector = ida_price_vector

        p_limit = 1.0  # kW
        storage_capacity = 4.0  # kWh
        p_charge_max = 1.0  # kW
        p_discharge_max = 1.0  # kW
        number_of_cycles = 1.0  # cycles per day
        efficiency = 1.0  # round-trip efficiency
        start_soc = 0.0  # 0% initial state of charge
        end_soc = 0.0  # 0% final state of charge

        # Get schedules
        pv_output, daa_price, p_charge_daa, p_discharge_daa, soc_daa, p_curtailed_daa, injection_power_daa = get_daa_schedule(pv_vector,
                                                                                                  daa_price_vector, 
                                                                                                  p_limit, 
                                                                                                  storage_capacity, 
                                                                                                  p_charge_max, 
                                                                                                  p_discharge_max, 
                                                                                                  number_of_cycles, 
                                                                                                  efficiency, 
                                                                                                  start_soc, 
                                                                                                  end_soc)

        ida_price, p_charge_ida, p_discharge_ida, p_close_charge_daa, p_close_discharge_daa, p_curtailed_ida, \
        p_close_curtailed_daa, p_curtailed_daa_ida ,p_charge_daa_ida, p_discharge_daa_ida, soc_ida, injection_power_ida = get_ida_schedule(
                                                                                                                        ida_price_vector, 
                                                                                                                        pv_output,
                                                                                                                        p_limit, 
                                                                                                                        storage_capacity, 
                                                                                                                        p_charge_max, 
                                                                                                                        p_discharge_max,
                                                                                                                        p_charge_daa,
                                                                                                                        p_discharge_daa,
                                                                                                                        p_curtailed_daa,
                                                                                                                        number_of_cycles, 
                                                                                                                        efficiency, 
                                                                                                                        start_soc, 
                                                                                                                        end_soc)
    
        idc_price, p_charge_idc, p_discharge_idc, p_close_charge_daa_ida, p_close_discharge_daa_ida, p_curtailed_idc, \
        p_close_curtailed_daa_ida, p_curtailed_daa_ida_idc, p_charge_daa_ida_idc, p_discharge_daa_ida_idc, soc_idc, injection_power_idc = get_idc_schedule(    
                                                                                                                                idc_price_vector,
                                                                                                                                pv_output, 
                                                                                                                                p_limit, 
                                                                                                                                storage_capacity, 
                                                                                                                                p_charge_max, 
                                                                                                                                p_discharge_max,
                                                                                                                                p_charge_daa_ida,
                                                                                                                                p_discharge_daa_ida,
                                                                                                                                p_curtailed_daa_ida,
                                                                                                                                number_of_cycles, 
                                                                                                                                efficiency, 
                                                                                                                                start_soc, 
                                                                                                                                end_soc
                                                                                                                            )
        # Compile results into a DataFrame
        results = pd.DataFrame({
                            # prices and PV output
                           'daa_price': daa_price,
                           'ida_price': ida_price,
                           'idc_price': idc_price,
                           'pv_output': pv_output,
                            # DAA results
                           'p_charge_daa': p_charge_daa,
                           'p_discharge_daa': p_discharge_daa,
                           'p_curtailed_daa': p_curtailed_daa,
                           'soc_daa': soc_daa,
                           'injection_power_daa': injection_power_daa,
                           # IDA results
                           'p_charge_ida': p_charge_ida,
                           'p_discharge_ida': p_discharge_ida,
                           'p_close_charge_daa': p_close_charge_daa,
                           'p_close_discharge_daa': p_close_discharge_daa,
                           'p_curtailed_ida': p_curtailed_ida,
                           'p_close_curtailed_daa': p_close_curtailed_daa,
                           'p_charge_daa_ida': p_charge_daa_ida,
                           'p_discharge_daa_ida': p_discharge_daa_ida,
                           'p_curtailed_daa_ida': p_curtailed_daa_ida,
                           'soc_ida': soc_ida,
                           'injection_power_ida': injection_power_ida,
                           # IDC results
                           'p_charge_idc': p_charge_idc,
                           'p_discharge_idc': p_discharge_idc,
                           'p_close_charge_daa_ida': p_close_charge_daa_ida,
                           'p_close_discharge_daa_ida': p_close_discharge_daa_ida,
                           'p_curtailed_idc': p_curtailed_idc,
                           'p_close_curtailed_daa_ida': p_close_curtailed_daa_ida,
                           'p_charge_daa_ida_idc': p_charge_daa_ida_idc,
                           'p_discharge_daa_ida_idc': p_discharge_daa_ida_idc,
                           'p_curtailed_daa_ida_idc': p_curtailed_daa_ida_idc,
                           'soc_idc': soc_idc,
                           'injection_power_idc': injection_power_idc
                          }
                         )

        ##### Calculate profits
        # Calculate annual profit from PV Output multiplied with DAA prices as reference
        results['pv_direct_market_profit_reference'] = results.daa_price * results.pv_output * 15 / 60
        # Only sum up profit at positive market prices, because plant will be curtailed at other times
        condition_daa_positive = results.daa_price >= 0
        # annual_pv_production_excl_negative_market_price_hours = results.loc[condition_daa_positive, 'pv_output'].sum() * 15 / 60
        annual_pv_direct_market_profit_reference = results.loc[condition_daa_positive, 'pv_direct_market_profit_reference'].sum()

        # Calculate annual profit on DAA market
        results['injected_power_direct_market_profit_daa'] = results.daa_price * results.injection_power_daa * 15 / 60
        annual_injected_energy_direct_market_profit_daa = results.injected_power_direct_market_profit_daa.sum()

        # Calculate annual profit on IDA market
        results['ida_market_profit'] = (results['p_discharge_ida'] - results['p_charge_ida'] 
                                        + results['p_close_charge_daa'] - results['p_close_discharge_daa'] 
                                        + results['p_close_curtailed_daa'] - results['p_curtailed_ida']) * results['ida_price']  / 4
        annual_ida_market_profit = results['ida_market_profit'].sum()

        # Calculate annual profit on IDC market
        results['idc_market_profit'] = (results['p_discharge_idc'] - results['p_charge_idc'] 
                                        + results['p_close_charge_daa_ida'] - results['p_close_discharge_daa_ida'] 
                                        + results['p_close_curtailed_daa_ida'] - results['p_curtailed_idc']) * results['idc_price']  / 4
        annual_idc_market_profit = results['idc_market_profit'].sum()
        
        # Assertions
        self.assertEqual(annual_injected_energy_direct_market_profit_daa, 4.0)
        self.assertEqual(annual_ida_market_profit, 2.0)
        self.assertEqual(annual_idc_market_profit, 0.0)
        self.assertEqual(results['p_curtailed_daa'].sum() / 4, 1.0)
        self.assertEqual(results['p_curtailed_daa_ida_idc'].sum() / 4, 0.0)