import tensorflow as tf

from datasets.IOT_DNL.iot_dnl import IOT_DNL
from datasets.TON_IOT.ton_iot import TON_IOT
from datasets.UNSW.unsw import UNSW
from datasets.Slicing5G.slicing5g import Slicing5G
from datasets.NetSlice5G.netslice5g import NetSlice5G
from datasets.Botnet_IOT.botnet_iot import Botnet_IOT
from datasets.IoTID20.iotid20 import IoTID20
from datasets.KPI_KQI.kpi_kqi import KPI_KQI
from datasets.QoS_QoE.qos_qoe import QoS_QoE
from datasets.RT_IOT.rt_iot import RT_IOT
from datasets.UNAC.unac import UNAC

DATASETS = {
    "NetSlice5G" : NetSlice5G,
    "Slicing5G": Slicing5G,
    "IOT_DNL": IOT_DNL,
    "UNSW": UNSW,
    "TON_IOT": TON_IOT,
    "Botnet_IOT" : Botnet_IOT,
    "IoTID20" : IoTID20,
    "KPI_KQI" : KPI_KQI,
    "QoS_QoE" : QoS_QoE,
    "RT_IOT" : RT_IOT,
    "UNAC" : UNAC
}