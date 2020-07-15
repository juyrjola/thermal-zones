import click
import time
import os
import re
import glob
from functools import cached_property

THERMAL_PATH = '/sys/devices/virtual/thermal/'
RAPL_PATH = '/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0'
POWERCAP_PATH = '/sys/devices/virtual/powercap'


def get_thermal_zone_paths():
    paths = [p for p in os.listdir(THERMAL_PATH) if re.match('thermal_zone[0-9]+', p)]
    return [os.path.join(THERMAL_PATH, p) for p in paths]


def read_sysfs_value(path):
    f = open(path, 'r', encoding='ascii')
    return f.read().strip()


def write_sysfs_value(path, val):
    f = open(path, 'w', encoding='ascii')
    return f.write(val.strip())


class SysfsMixin:
    def read_attr(self, attr):
        return read_sysfs_value(os.path.join(self.base_path, attr))

    def write_attr(self, attr, val):
        return write_sysfs_value(os.path.join(self.base_path, attr), val)


class Constraint(SysfsMixin):
    def __init__(self, base_path, nr):
        self.base_path = base_path
        self.nr = nr
        self._power_limit_uw = self.power_limit_uw
        self._power_limit_changed = False

    def restore(self):
        if not self._power_limit_changed:
            return
        print('%s [%s]: %s -> %s' % (self.base_path, self.name, self.power_limit_uw, self._power_limit_uw))
        self.set_power_limit_uw(self._power_limit_uw)

    def read_attr(self, attr):
        return super().read_attr('constraint_%d_%s' % (self.nr, attr))

    def write_attr(self, attr, val):
        return super().write_attr('constraint_%d_%s' % (self.nr, attr), val)

    @property
    def name(self):
        return self.read_attr('name')

    @property
    def max_power_uw(self):
        try:
            out = int(self.read_attr('max_power_uw'))
        except OSError:
            return None
        return out

    @property
    def max_power(self):
        if self.max_power_uw is None:
            return None
        return self.max_power_uw / 1000000

    @property
    def power_limit_uw(self):
        return int(self.read_attr('power_limit_uw'))

    def set_power_limit_uw(self, val):
        self._power_limit_changed = True
        self.write_attr('power_limit_uw', str(int(val)))

    @property
    def power_limit(self):
        return self.power_limit_uw / 1000000

    @property
    def time_window_us(self):
        return int(self.read_attr('time_window_us'))

    def __str__(self):
        return self.name


class Battery(SysfsMixin):
    def __init__(self):
        self.base_path = '/sys/class/power_supply/BAT1'

    @property
    def power(self):
        return int(self.read_attr('power_now')) / 1000000


class CPU:
    def __init__(self, path):
        self.path = path
        self.nr = int(path[-1])
        self.max_freq = int(self.read_cpufreq('cpuinfo_max_freq'))
        self.min_freq = int(self.read_cpufreq('cpuinfo_min_freq'))
        self._scaling_max_freq = self.scaling_max_freq
        self._ep_pref = self.energy_performance_preference
        self._scaling_gov = self.scaling_governor

    def init(self):
        # 'power', 'balance_power', 'balance_performance', 'performance'
        self.set_energy_performance_preference('power')
        # 'performance', 'powersave'
        self.set_scaling_gov('powersave')
        self.set_scaling_max_freq(self.max_freq)

    def restore(self):
        self.set_energy_performance_preference(self._ep_pref)
        self.set_scaling_gov(self._scaling_gov)

    @property
    def energy_performance_preference(self):
        return self.read_cpufreq('energy_performance_preference')

    def set_energy_performance_preference(self, pref):
        if self.energy_performance_preference != pref:
            self.write_cpufreq('energy_performance_preference', pref)

    @property
    def scaling_max_freq(self):
        return int(self.read_cpufreq('scaling_max_freq'))

    def set_scaling_max_freq(self, freq):
        if self.scaling_max_freq != freq:
            self.write_cpufreq('scaling_max_freq', str(freq))

    @property
    def scaling_governor(self):
        return self.read_cpufreq('scaling_governor')

    def set_scaling_gov(self, gov):
        if self.scaling_governor != gov:
            self.write_cpufreq('scaling_governor', gov)

    @property
    def cur_freq(self):
        return self.read_cpufreq('scaling_cur_freq')

    def read_attr(self, attr):
        return read_sysfs_value(os.path.join(self.path, attr))

    def write_attr(self, attr, val):
        return write_sysfs_value(os.path.join(self.path, attr), val)

    def read_cpufreq(self, attr):
        return self.read_attr('cpufreq/%s' % attr)

    def write_cpufreq(self, attr, val):
        print('[CPU%d] Setting %s to %s' % (self.nr, attr, val))
        return self.write_attr('cpufreq/%s' % attr, val)


class PowerCapDevice:
    def __init__(self, path):
        self.path = path
        self.constraints = []
        self._enabled = self.enabled
        self._enabled_changed = False
        self._last_energy_sample_time = None
        self._find_constraints()
        self.print()

    def restore(self):
        for c in self.constraints:
            c.restore()
        if not self._enabled_changed:
            return
        print('%s: %s -> %s' % (self.name, self.enabled, self._enabled))
        self.set_enabled(self._enabled)

    def read_attr(self, attr):
        return read_sysfs_value(os.path.join(self.path, attr))

    def write_attr(self, attr, val):
        return write_sysfs_value(os.path.join(self.path, attr), val)

    def _find_constraints(self):
        for fn in os.listdir(self.path):
            m = re.match('constraint_([0-9]+)_name', fn)
            if not m:
                continue
            self.constraints.append(Constraint(self.path, int(m.groups()[0])))

    @property
    def enabled(self):
        return bool(int(self.read_attr('enabled')))

    def set_enabled(self, val: bool):
        self._enabled_changed = True
        self.write_attr('enabled', '1' if val else '0')

    @property
    def name(self):
        return self.read_attr('name')

    @property
    def power(self):
        energy_uj = int(self.read_attr('energy_uj'))
        now = time.time()
        if self._last_energy_sample_time is not None:
            power = (energy_uj - self._last_energy_uj) / (now - self._last_energy_sample_time) / 1000000
        else:
            power = 0
        self._last_energy_sample_time = now
        self._last_energy_uj = energy_uj
        return power

    def set_power_limit(self, limit_mw):
        if limit_mw is None:
            print('%s: restoring' % self)
            self.restore()
            return

        print('%s: limit to %.3f W' % (self.name, limit_mw / 1000))

        for c in self.constraints:
            if c.name == 'short_term':
                break

        c.set_power_limit_uw(limit_mw * 1000)
        print(c.power_limit_uw)
        if not self.enabled:
            self.set_enabled(True)

        self.print()

    def print(self):
        print('%s [%s] %s' % (self.name, 'enabled' if self.enabled else 'disabled', self.path))
        for c in self.constraints:
            print('  %s (limit: %.3f, max: %s)' % (c.name, c.power_limit, c.max_power))


class PowerCap:
    NO_LIMIT = 0
    HOT_LIMIT = 1
    CRITICAL_LIMIT = 2

    def __init__(self, base_path=POWERCAP_PATH):
        self.base_path = base_path
        self.devices = []
        self._scan()

    def restore(self):
        for d in self.devices:
            d.restore()

    def set_power_limit(self, name, limit):
        found = False
        for d in self.devices:
            if d.name == name:
                d.set_power_limit(limit)
                found = True
        if not found:
            raise Exception('Unknown cap device: %s' % name)

    def set_limit(self, limit):
        if limit == self.NO_LIMIT:
            self.restore()
        elif limit == self.HOT_LIMIT:
            self.set_power_limit('package-0', 16000)
            self.set_power_limit('core', 6000)
        elif limit == self.CRITICAL_LIMIT:
            self.set_power_limit('package-0', 8000)
            self.set_power_limit('core', 2000)

    def _find_devices(self, path):
        for fname in os.listdir(path):
            if fname == 'energy_uj':
                self.devices.append(PowerCapDevice(path))
                continue

            p = self.subpath(path, fname)
            if os.path.islink(p):
                continue
            if os.path.isdir(p):
                self._find_devices(p)
                continue

    def subpath(self, *paths):
        return os.path.join(self.base_path, *paths)

    def _scan(self, base_path=None):
        for p in os.listdir(self.base_path):
            if not os.path.isdir(self.subpath(p)):
                continue
            if os.path.exists(self.subpath(p, 'enabled')):
                if read_sysfs_value(self.subpath(p, 'enabled')) != '1':
                    print('Disabled: %s' % p)
                    continue

            self._find_devices(self.subpath(p))


class TripPoint:
    def __init__(self, zone, path):
        self.zone = zone
        self.path = path

    def read_attr(self, attr):
        return read_sysfs_value(os.path.join(self.path + '_' + attr))

    @cached_property
    def temp(self):
        return int(self.read_attr('temp'))

    @property
    def temp_c(self):
        return self.temp / 1000

    @cached_property
    def type(self):
        return self.read_attr('type')


class ThermalZone:
    def __init__(self, path):
        self.path = path
        tp_paths = glob.glob(os.path.join(self.path, 'trip_point_*_type'))
        tps = [TripPoint(self, p.replace('_type', '')) for p in tp_paths]
        self.trip_points = sorted(filter(lambda x: x.temp > 0, tps), key=lambda x: x.temp, reverse=True)
        try:
            self.last_state = self.get_current_state()
        except OSError:
            self.valid = False
            return

        self.first_tp = self.trip_points[-1] if self.trip_points else None
        if self.first_tp:
            self.hot_tp = list(filter(lambda x: x.type in ('hot', 'critical'), self.trip_points))[-1]
        self.valid = True

    def __str__(self):
        return self.type

    def read_attr(self, attr):
        return read_sysfs_value(os.path.join(self.path, attr))

    @property
    def type(self):
        return self.read_attr('type')

    @property
    def temp(self):
        return int(self.read_attr('temp'))

    @property
    def temp_c(self):
        return self.temp / 1000

    def get_scaled_temp(self):
        if not self.trip_points:
            return 0.0

        current = self.temp
        first = self.first_tp.temp
        last = self.hot_tp.temp

        if current <= first:
            return 0.0
        if current >= last:
            return 1.0
        ret = (current - first) / (last - first)
        return ret

    def get_current_state(self):
        temp = self.temp
        for tp in self.trip_points:
            if temp >= tp.temp:
                return tp.type
        return None

    def state_changed(self) -> bool:
        state = self.get_current_state()
        if state != self.last_state:
            self.last_state = state
            return True
        return False


class ThermalDaemon:
    def __init__(self):
        self.pc = PowerCap()
        tzs = [ThermalZone(p) for p in get_thermal_zone_paths()]
        self.thermal_zones = [t for t in tzs if t.valid]

        self.battery = Battery()

        self.cpus = []
        cpu_paths = glob.glob('/sys/devices/system/cpu/cpu?')
        for p in sorted(cpu_paths):
            self.cpus.append(CPU(p))

        for z in self.thermal_zones:
            print('%s: %.1f %s (%s)' % (z.type, z.temp / 1000, z.get_current_state(), z.path))
            for tp in sorted(z.trip_points, key=lambda x: x.temp):
                print('  %.1f: %s' % (tp.temp / 1000, tp.type))

        self.last_state = None
        self.callback_func = None

    def init(self):
        CPUIDLE_GOV = 'teo'
        CPUIDLE_PATH = '/sys/devices/system/cpu/cpuidle/current_governor'
        gov = read_sysfs_value(CPUIDLE_PATH)
        if gov != CPUIDLE_GOV:
            print('Setting cpuidle governor to %s' % CPUIDLE_GOV)
            write_sysfs_value(CPUIDLE_PATH, CPUIDLE_GOV)

        for cpu in self.cpus:
            cpu.init()

        self.loop_count = 0

    def restore(self):
        self.pc.restore()
        for cpu in self.cpus:
            cpu.restore()

    def run(self):
        self.loop_count = 0
        while True:
            try:
                self.print_state()
                self.loop()
                time.sleep(2)
                self.loop_count += 1
            except (Exception, KeyboardInterrupt):
                print('restoring')
                self.restore()
                raise

    def set_state(self, state):
        if state is None or state == 'active':
            self.pc.set_limit(PowerCap.NO_LIMIT)
        elif state == 'passive':
            self.pc.set_limit(PowerCap.HOT_LIMIT)
        else:
            self.pc.set_limit(PowerCap.CRITICAL_LIMIT)
        self.last_state = state

    def print_state(self):
        s1 = ''
        s2 = ''
        s3 = ''
        s4 = ''
        for z in self.thermal_zones:
            s1 += '%-15s' % z.type
            s2 += '%-15.1f' % (z.temp / 1000)
            s3 += '%-15s' % z.get_current_state()
            s4 += '%-15d' % (z.get_scaled_temp() * 100)

        if self.loop_count % 20 == 0:
            print(s1)
            print(s3)
        print(s2)
        print(s4)

        for d in self.pc.devices:
            print('%s: %.02f W [%s]' % (d.name, d.power, d.path))
        print('Battery: %.02f W' % self.battery.power)

    def loop(self):
        worst_state = None
        for tz in self.thermal_zones:
            state = tz.get_current_state()
            if tz.state_changed():
                print('%s changed to %s' % (str(tz), state))

            if state is None:
                continue
            if worst_state is None:
                worst_state = state
                continue
            if state == 'critical':
                worst_state = state
                continue
            if state == 'hot' and worst_state != 'hot':
                worst_state = state
                continue
            if state == 'passive' and worst_state == 'active':
                worst_state = state
                continue

        if worst_state != self.last_state:
            # self.pc.limit(worst_state)
            print('state change to %s' % worst_state)
            self.set_state(worst_state)

        if self.callback_func:
            self.callback_func()

    def set_callback(self, callback_func):
        self.callback_func = callback_func


class Plotter:
    def __init__(self, d: ThermalDaemon):
        self.d = d


class MatplotPlotter(Plotter):
    def init(self):
        import matplotlib
        # matplotlib.use('GTK3Cairo')
        matplotlib.use('GTK3Cairo')
        import matplotlib.pyplot as plt

        self.plt = plt
        fig, axs = plt.subplots(2)
        self.fig = fig
        ax = axs[0]
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 120)

        for tz in self.d.thermal_zones:
            if tz.type in ('acpitz', 'INT3400 Thermal', 'x86_pkg_temp'):
                tz.line = None
                continue
            ydata = [tz.temp_c] * 50
            tz.line, = ax.plot(ydata, label=tz.type)

        ax.legend()

        ax = axs[1]
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 20)

        bat = self.d.battery
        bat.line, = ax.plot([bat.power] * 50, label='Battery')

        for dev in self.d.pc.devices:
            dev.line, = ax.plot([dev.power] * 50, label=dev.name)

        ax.legend()
        plt.show(block=False)
        plt.pause(0.5)

    def update(self, frame=None):
        self.d.loop()

        lines = []

        for tz in self.d.thermal_zones:
            if not tz.line:
                continue
            data = list(tz.line.get_ydata())[1:]
            data.append(tz.temp_c)
            tz.line.set_ydata(data)
            lines.append(tz.line)

        bat = self.d.battery
        data = list(bat.line.get_ydata())[1:]
        data.append(bat.power)
        bat.line.set_ydata(data)
        lines.append(bat.line)

        for dev in self.d.pc.devices:
            data = list(dev.line.get_ydata())[1:]
            data.append(dev.power)
            dev.line.set_ydata(data)
            lines.append(dev.line)

        return lines

    def run(self):
        from matplotlib.animation import FuncAnimation

        ani = FuncAnimation(self.fig, self.update, frames=None, blit=True, interval=500)  # noqa
        print('showing')
        self.plt.show()


class QTGraphPlotter(Plotter):
    def init(self):
        from pyqtgraph.Qt import QtGui, QtCore
        import pyqtgraph as pg
        import signal
        import seaborn as sns

        QtGui.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtGui.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

        self.app = app = QtGui.QApplication([])
        app.setStyle("fusion")

        self.win = win = pg.GraphicsLayoutWidget(show=True, title='Thermal')
        win.resize(1000, 600)
        win.setWindowTitle('Thermal Plot')

        palette = sns.color_palette('deep')

        pg.setConfigOptions(antialias=True)
        p = self.temp_plot = win.addPlot(title='Temps')
        p.setLabel('left', '°C')
        p.addLegend()
        p.setMouseEnabled(x=False, y=False)
        ci = 0
        for tz in self.d.thermal_zones:
            if tz.type in ('acpitz', 'INT3400 Thermal', 'x86_pkg_temp'):
                tz.line = None
                continue
            color = [x * 255 for x in palette[ci]]
            tz.line = p.plot(pen=pg.mkPen(color, width=4), name=tz.type)
            ci += 1

        p.setYRange(0, 120, padding=0)
        p.setXRange(0, 50, padding=0)
        p.enableAutoRange('xy', False)

        win.nextRow()

        p = self.power_plot = win.addPlot(title='Power')
        p.setLabel('left', 'W')
        p.addLegend()
        p.setMouseEnabled(x=False, y=False)
        p.setYRange(0, 30, padding=0)
        p.setXRange(0, 50, padding=0)
        p.enableAutoRange('xy', False)

        ci = 0
        bat = self.d.battery
        color = [x * 255 for x in palette[ci]]
        bat.line = p.plot(pen=pg.mkPen(color, width=4), name='Battery')
        ci += 1
        for dev in self.d.pc.devices:
            color = [x * 255 for x in palette[ci]]
            dev.line = p.plot(pen=pg.mkPen(color, width=4), name=dev.name)
            ci += 1

        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, signum, frame):
        self.app.quit()

    def update_line(self, line, sample):
        x, y = line.getData()
        if y is None:
            y = []
        else:
            y = list(y)
        y.append(sample)
        if len(y) > 50:
            y = y[1:]
        line.setData(y)

    def _update(self):
        self.d.loop()
        for tz in self.d.thermal_zones:
            if not tz.line:
                continue
            self.update_line(tz.line, tz.temp_c)

        bat = self.d.battery
        self.update_line(bat.line, bat.power)

        for dev in self.d.pc.devices:
            self.update_line(dev.line, dev.power)

    def update(self):
        try:
            self._update()
        except Exception:
            print('quitting')
            self.app.quit()
            raise

    def safe_timer(self):
        from pyqtgraph.Qt import QtCore

        self.update()
        QtCore.QTimer.singleShot(2000, self.safe_timer)

    def run(self):
        self.safe_timer()
        self.app.exec_()


@click.command()
@click.option('--plot', is_flag=True)
def run(plot):
    d = ThermalDaemon()
    if plot:
        plotter = QTGraphPlotter(d)
        plotter.init()

    d.init()

    if plot:
        try:
            plotter.run()
        finally:
            print('restoring')
            d.restore()
    else:
        d.run()


if __name__ == "__main__":
    run()
