from setuptools import setup

setup(    
    name="Pawlicy",
    version='2.0.0',
    install_requires=[
        'gym==0.21',
        'pybullet',
        'numpy',
        'matplotlib',
        'stable-baselines3[extra]',
        'perlin-noise',
        'pyyaml'
    ]
)