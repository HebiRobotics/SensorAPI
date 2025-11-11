"""
(optional) conda init
conda create -n TactX

conda activate TactX


# Install dependencies
pip install pyserial numpy

# Direct scan, default baud rate 921600, 20x8 matrix, 2 bytes per pixel, single port detection for 3 seconds
python 000_Sensor_Connection.py

# Or customize parameters (e.g., probe longer)
python 000_Sensor_Connection.py --probe 5.0

# If your device is not using 921600 baud rate
python 000_Sensor_Connection.py --baud 460800

Serial port names on different systems:
Windows: "COM3"
Ubuntu: "/dev/ttyUSB0"
"""
import argparse
import time
from typing import Optional

import numpy as np
import serial
from serial.tools import list_ports

HEADER = bytes([0xFF, 0xFF])

def detect_one_port(
    port: str,
    baud: int,
    num_rows: int,
    num_cols: int,
    bytes_per_pixel: int,
    probe_seconds: float,
    read_chunk: int = 512,
    serial_timeout: float = 0.01,
) -> bool:
    """
    Open the specified serial port and attempt to detect one valid frame within `probe_seconds`.
    Frame definition: HEADER(2) + payload(num_rows * num_cols * bytes_per_pixel)
    Returns True if a valid frame is detected.
    """
    payload_size = num_rows * num_cols * bytes_per_pixel
    total_size = len(HEADER) + payload_size

    try:
        with serial.Serial(port, baudrate=baud, timeout=serial_timeout) as ser:
            ser.reset_input_buffer()
            t_end = time.perf_counter() + probe_seconds
            buf = bytearray()

            while time.perf_counter() < t_end:
                chunk = ser.read(read_chunk)
                if chunk:
                    buf += chunk

                    # Control buffer size to prevent unlimited growth
                    if len(buf) > max(4096, 4 * total_size):
                        del buf[: -2048]

                    # Try to find header and parse one frame
                    while True:
                        start = buf.find(HEADER)
                        if start < 0:
                            # Keep only potential header prefix
                            if len(buf) > len(HEADER) - 1:
                                del buf[: - (len(HEADER) - 1)]
                            break

                        if len(buf) - start < total_size:
                            # Not enough bytes yet, wait for more data
                            # Discard useless bytes before the header
                            if start > 0:
                                del buf[:start]
                            break

                        # Extract payload
                        frame_start = start + len(HEADER)
                        frame_end = start + total_size
                        payload = bytes(buf[frame_start: frame_start + payload_size])

                        # Try parsing as >u2 (same format as your existing ForceSensor)
                        try:
                            arr = np.frombuffer(payload, dtype=">u2").reshape(num_rows, num_cols)
                            # If parsing succeeds, this port is a valid sensor port
                            return True
                        except Exception:
                            # Invalid frame, skip one byte and continue looking for the next possible header
                            del buf[: start + 1]
                            continue

                        # Normally should not reach here
                else:
                    # No data, sleep briefly
                    time.sleep(0.001)

    except (serial.SerialException, OSError):
        return False

    return False


def main():
    parser = argparse.ArgumentParser(description="Scan serial ports and detect ForceSensor.")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--rows", type=int, default=20, help="Number of rows (default: 20)")
    parser.add_argument("--cols", type=int, default=8, help="Number of columns (default: 8)")
    parser.add_argument("--bpp", type=int, default=2, help="Bytes per pixel (default: 2)")
    parser.add_argument("--probe", type=float, default=3.0, help="Probe duration per port in seconds (default: 3s)")
    parser.add_argument("--timeout", type=float, default=0.01, help="Serial read timeout (default: 0.01s)")
    args = parser.parse_args()

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("Found ports:")
    for p in ports:
        print(f"  - {p.device} ({p.description})")

    print("\nProbing ports...\n")
    good = []
    for p in ports:
        dev = p.device
        ok = detect_one_port(
            port=dev,
            baud=args.baud,
            num_rows=args.rows,
            num_cols=args.cols,
            bytes_per_pixel=args.bpp,
            probe_seconds=args.probe,
            serial_timeout=args.timeout,
        )
        print(f"[{dev}] {'OK (sensor detected)' if ok else 'NO DATA / NOT MATCH'}")
        if ok:
            good.append(dev)

    print("\nSummary:")
    if good:
        for g in good:
            print(f"  * {g}  <-- sensor present")
    else:
        print("  No sensor detected on any port.")

if __name__ == "__main__":
    main()
