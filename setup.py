from setuptools import setup


setup(
    name='qiskit_tricks',
    packages=[
        'qiskit_tricks',
        'qiskit_tricks.experiments',
        'qiskit_tricks.fit',
    ],
    python_requires='>=3.9',
    install_requires=[
        'qiskit',
        'qiskit_experiments',
        'numpy',
        'pandas',
        'injector',
    ],
)