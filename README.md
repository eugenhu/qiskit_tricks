# qiskit_tricks

Not an official Qiskit library. Just a collection of convenience functions to use with Qiskit.

The most useful functions are just `resultdf()` and `parallelize_circuits()`.

See the 'examples' folder for complete examples.

## Demo
1. [resultdf()](#resultdf)
2. [parallelize_circuits()](#parallelize_circuits)
3. [bake_schedule()](#bake_schedule)
4. [get_play_instruction()](#get_play_instruction)
5. [install_parametric_pulse()](#install_parametric_pulse)

### `resultdf()`
Convert a Qiskit `Result` to a pandas series or frame.
```python
>>> resultdf(job.result())
                                            count
job_id                               state       
05df4f33-2a3b-42de-805f-b350edab3dd1 1        520
                                     0        504
```

### `parallelize_circuits()`
Combine multiple circuits acting on disjoint qubits into one circuit.
```python
>>> q = QuantumRegister(2, 'q')
>>> c = ClassicalRegister(1, 'c')
>>> circ1 = QuantumCircuit(q, c)
>>> circ1.id(q[0]);
>>> circ1.measure(q[0], c[0]);
>>> circ1.draw()
     ┌───┐┌─┐
q_0: ┤ I ├┤M├
     └───┘└╥┘
q_1: ──────╫─
           ║ 
c: 1/══════╩═
           0
>>> circ2 = QuantumCircuit(q, c)
>>> circ2.x(q[1])
>>> circ2.measure(q[1], c[0])
>>> circ2.draw()
            
q_0: ────────
     ┌───┐┌─┐
q_1: ┤ X ├┤M├
     └───┘└╥┘
c: 1/══════╩═
           0
>>> circ = parallelize_circuits([circ1, circ2])[0]
>>> circ.draw()
      ┌───┐┌─┐   
 q_0: ┤ I ├┤M├───
      ├───┤└╥┘┌─┐
 q_1: ┤ X ├─╫─┤M├
      └───┘ ║ └╥┘
c0: 1/══════╩══╬═
            0  ║ 
               ║ 
c1: 1/═════════╩═
               0 
```
The combined circuit metadata contains information for `resultdf()` to "demultiplex" the results to appear as if each subcircuit was run individually.
```python
>>> circ.metadata
{'subcircuits': [{'name': 'circuit-1', 'creg_map': (('c0', 'c'),)},
  {'name': 'circuit-2', 'creg_map': (('c1', 'c'),)}]}
```

### `bake_schedule()`
Bake a schedule with multiple pulses into a schedule with a single Waveform with pre/post phase shifts. Mainly used to circumvent minimum pulse length constraints.
```python
>>> d0 = DriveChannel(0)
>>> sched = Schedule(
>>>     (0, ShiftPhase(pi, d0)),
>>>     (0, Play(Waveform([1+0j, 1+0j]), d0)),
>>>     (2, ShiftPhase(pi/2, d0)),
>>>     (2, Play(Waveform([1+0j, 1+0j]), d0)),
>>> )
>>> bake_schedule(sched)
ScheduleBlock(
  ShiftPhase(3.141592653589793, DriveChannel(0)),
  Play(Waveform(array([1.000000e+00+0.j, 1.000000e+00+0.j, 6.123234e-17+1.j, 6.123234e-17+1.j])), DriveChannel(0)),
  ShiftPhase(1.5707963267948966, DriveChannel(0)),
  name="block0",
  transform=AlignSequential()
)
```

### `get_play_instruction()`
Return the first Play instruction in a schedule.
```python
>>> inst_map = FakeArmonk().defaults().instruction_schedule_map
>>> get_play_instruction(inst_map.get('x', 0))
Play(Drag(duration=320, amp=(0.8183699822712108+0j), sigma=80, beta=-0.6793150565689698, name='drag_f7ce'), DriveChannel(0), name='drag_f7ce')
```

### `install_parametric_pulse()`
Monkey patch install a new parametric pulse into `qiskit.assembler.assemble_schedules`. I'm not sure if there's a proper way to do this.
```python
>>> class Cosine(ParametricPulse):
...     def __init__(self, duration, amp, name=None, limit_amplitude=None):
...         self.amp = amp
...         super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)
...
...     def get_waveform(self):
...         x = np.linspace(-.5, .5, self.duration)
...         return Waveform(self.amp * (0.5 + 0.5*np.cos(2*np.pi*x)))
...
...     def validate_parameters(self):
...         pass
...
...     @property
...     def parameters(self):
...         return {'duration': self.duration, 'amp': self.amp}
...
...     def __repr__(self):
...         return f"{__class__.__name__}(duration={self.duration}, amp={self.amp})"
>>> sched = Schedule(Play(Cosine(160, 0.2), DriveChannel(0)))
>>> qiskit.compiler.assemble(sched, backend)
ValueError: <class '__main__.Cosine'> is not a valid ParametricPulseShapes
>>> install_parametric_pulse('cosine', Cosine)
>>> qiskit.compiler.assemble(sched, backend)
PulseQobj(...)
```
This is a hack, see the source code for implementation.
