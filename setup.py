from setuptools import setup

setup(
    name='spk',
    version='0.0.1',
    entry_points = {
        'console_scripts': ['spk=sparkle.src.core.main:main',
                            'bench_pex=sparkle.src.bench.pex:main']
    }
)
