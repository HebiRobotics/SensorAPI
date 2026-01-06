import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from TactXAPI.ForceSensor import ForceSensor
from TactXAPI.MultiSensor import MultiSensor


class SensorPlotter:
    def __init__(self):
        # Initialize sensors
        self.left_sensor = ForceSensor("/dev/ttyUSB0", baud_rate=921600)
        self.right_sensor = ForceSensor("/dev/ttyUSB1", baud_rate=921600)

        # MultiSensor setup with 100 Hz read and emit rates
        self.ms = MultiSensor(
            sensors={"left": self.left_sensor, "right": self.right_sensor},
            read_hz=100.0,
            emit_hz=100.0,
        )

        # Store latest frames (convert from milliNewtons to Newtons)
        self.left_frame = np.zeros((20, 8), dtype=np.float32)
        self.right_frame = np.zeros((20, 8), dtype=np.float32)
        self.timestamp = 0.0

        # Force threshold parameters
        self.force_threshold = 0.05  # Threshold in Newtons to discard low values
        self.max_force_threshold = 4.095  # Maximum force threshold to detect spikes (N)

        # Set up the callback
        self.ms.on_emit = self.on_emit

        # Set up the GUI
        self.setup_gui()

    def on_emit(self, batch, ts):
        """Callback to update frames from MultiSensor"""
        # Convert from milliNewtons to Newtons (divide by 1,000)
        # Flip along horizontal axis (up-down)
        left_new = np.flipud(batch["left"].astype(np.float32) / 1e3)
        right_new = np.flipud(batch["right"].astype(np.float32) / 1e3)

        # Spike detection: check if max force exceeds threshold
        left_max = left_new.max()
        right_max = right_new.max()

        # Update left frame only if no spike detected
        if left_max <= self.max_force_threshold:
            self.left_frame = left_new
        else:
            print(f"Spike detected in left sensor: {left_max:.3f} N (ignored)")

        # Update right frame only if no spike detected
        if right_max <= self.max_force_threshold:
            self.right_frame = right_new
        else:
            print(f"Spike detected in right sensor: {right_max:.3f} N (ignored)")

        self.timestamp = ts

    def setup_gui(self):
        """Set up the matplotlib GUI with heatmaps"""
        self.fig = plt.figure(figsize=(14, 6))

        # Create two subplots for the heatmaps
        self.ax_left = self.fig.add_subplot(1, 2, 1)
        self.ax_right = self.fig.add_subplot(1, 2, 2)

        self.fig.suptitle('Tactile Sensor Feedback', fontsize=14, fontweight='bold')

        # Initial images for both sensors
        # Max value in Newtons (4095 mN = 4.095 N)
        self.im_left = self.ax_left.imshow(
            self.left_frame,
            cmap='hot',
            interpolation='nearest',
            vmin=0,
            vmax=4095 / 1e3  # Convert max mN to N
        )
        self.ax_left.set_title('Left Sensor')
        self.ax_left.axis('off')

        self.im_right = self.ax_right.imshow(
            self.right_frame,
            cmap='hot',
            interpolation='nearest',
            vmin=0,
            vmax=4095 / 1e3
        )
        self.ax_right.set_title('Right Sensor')
        self.ax_right.axis('off')

        # Add colorbar to show force scale
        self.fig.colorbar(self.im_right, ax=self.ax_right, label='Force (N)')

        # Text displays for max and average values
        self.text_left = self.ax_left.text(
            0.02, 0.98, '',
            transform=self.ax_left.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10
        )
        self.text_right = self.ax_right.text(
            0.02, 0.98, '',
            transform=self.ax_right.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10
        )

    def update_plot(self, frame_num):
        """Animation update function"""
        # Update left sensor heatmap
        self.im_left.set_data(self.left_frame)
        lv, lr, lc = self.ms.get_max("left")
        lv_n = lv / 1e3  # Convert mN to N

        # Calculate average force for left sensor (above threshold)
        left_filtered_avg = self.left_frame[self.left_frame >= self.force_threshold]
        left_avg = left_filtered_avg.mean() if len(left_filtered_avg) > 0 else 0.0

        self.text_left.set_text(f'Max: {lv_n:.3f} N @ ({lr}, {lc})\nAvg: {left_avg:.3f} N')

        # Update right sensor heatmap
        self.im_right.set_data(self.right_frame)
        rv, rr, rc = self.ms.get_max("right")
        rv_n = rv / 1e3  # Convert mN to N

        # Calculate average force for right sensor (above threshold)
        right_filtered_avg = self.right_frame[self.right_frame >= self.force_threshold]
        right_avg = right_filtered_avg.mean() if len(right_filtered_avg) > 0 else 0.0

        self.text_right.set_text(f'Max: {rv_n:.3f} N @ ({rr}, {rc})\nAvg: {right_avg:.3f} N')

        return self.im_left, self.im_right, self.text_left, self.text_right

    def run(self):
        """Start the sensors and GUI"""
        try:
            print("Starting sensor plotting at 100 Hz...")
            self.ms.start()

            # Set up animation (update GUI at ~30 FPS for smooth display)
            self.ani = FuncAnimation(
                self.fig,
                self.update_plot,
                interval=33,  # ~30 FPS
                blit=False,
                cache_frame_data=False
            )

            plt.show()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.close()

    def close(self):
        """Clean up resources"""
        print("Closing sensors...")
        self.ms.close()
        plt.close(self.fig)
        print("Stopped.")


def main():
    plotter = SensorPlotter()
    plotter.run()


if __name__ == "__main__":
    main()
