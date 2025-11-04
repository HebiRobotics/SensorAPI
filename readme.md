# API Use Example

This document provides examples for reading data from the **Touch Tronix tactile sensor**.  
Tested on **Ubuntu 22.04** and **Windows**.

---

## Requirements

Before running the examples, make sure you have installed the required package:

```
pip install pyserial
```


Examples
Example 1 — Check Available Serial Ports

File: 000_Sensor_Connection.py
This example identifies the available serial ports on your computer.

Run command:

```
python 000_Sensor_Connection.py
```



Example 2 — Read Single Sensor Data

File: 001_ForceSensor_Single.py
This example reads data from one tactile sensor and prints the readings in the terminal.

Run command:
```
python 001_ForceSensor_Single.py
```



Example 3 — Read Multiple Sensors with Adjustable Frequency

File: 002_MultiSensor_Frequency.py
This example allows the computer to read multiple sensors and adjust the output frequency.
(Default frequency: 100 Hz)

Run command:
```
python 002_MultiSensor_Frequency.py
```




✅ Tip:
Ensure your sensor is properly connected via USB/serial port before running any script.
You can check the connected devices using the following commands:

For Ubuntu:

ls /dev/tty*

make sure you have enable the premission on ubuntu
```
chmod a+rw /dev/tty0
```
replace the serial port if needed




Supported Platforms:

✅ Ubuntu 22.04

✅ Windows 10/11

