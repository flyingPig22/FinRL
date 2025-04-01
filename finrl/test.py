from __future__ import annotations

from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    net_dim,
    if_vix=True,
    **kwargs,
):
    # import data processor
    from finrl.meta.data_processor import DataProcessor
    import pandas as pd
    from datetime import timedelta
    start_date_timestamp = pd.Timestamp(start_date)
    end_date_timestamp = pd.Timestamp(end_date)
    start_date_buffer_timestamp = start_date_timestamp - timedelta(days=300)
    start_date_buffer = start_date_buffer_timestamp.strftime('%Y-%m-%d')
    # fetch data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date_buffer, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = dp.add_vix(data)
    data["tmp-timestamp"] = pd.to_datetime(data['timestamp'])
    history_data = data
    data = data[(data['tmp-timestamp'] >= start_date_timestamp) & (data['tmp-timestamp'] <= end_date_timestamp)]
    data.drop(columns=['tmp-timestamp'], inplace=True)

    last_row = data.iloc[-1]
    data = pd.concat([data, pd.DataFrame([last_row])], ignore_index=True)

    data = data.reset_index(drop=True)
    # price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    timestamp_array, price_array, tech_array, turbulence_array, volume_array, open_price_array, high_price_array, low_price_array = dp.custom_df_to_array(
        data, if_vix)
    history_timestamp_array, history_price_array, history_tech_array, history_turbulence_array, history_volume_array, history_open_price_array, history_high_price_array, history_low_price_array = dp.custom_df_to_array(
        history_data, if_vix)
    addition_days = len(history_price_array) - len(price_array)
    days_needed = 10
    final_history_price_array = history_price_array[addition_days - days_needed:]
    final_history_turbulence_array = history_turbulence_array[addition_days - days_needed:]
    final_history_volume_array = history_volume_array[addition_days - days_needed:]
    final_history_tech_array = history_tech_array[addition_days - days_needed:]

    env_config = {
        "timestamp_array": timestamp_array,
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
        "volume_array": volume_array,
        "open_price_array": open_price_array,
        "high_price_array": high_price_array,
        "low_price_array": low_price_array,
        "history_price_array": final_history_price_array,
        "history_turbulence_array": final_history_turbulence_array,
        "history_volume_array": final_history_volume_array,
        "history_tech_array": final_history_tech_array
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dim,
            environment=env_instance,
            env_args=env_config
        )
        return episode_total_assets
    elif drl_lib == "rllib":
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=cwd,
        )
        return episode_total_assets
    elif drl_lib == "stable_baselines3":
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )
        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        net_dimension=512,
        kwargs=kwargs,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # account_value_rllib = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo/checkpoint_000030/checkpoint-30",
    #     rllib_params=RLlib_PARAMS,
    # )
    #
    # # demo for stable baselines3
    # account_value_sb3 = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac.zip",
    # )
