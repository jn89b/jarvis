# Evader Dataset
- Contains the dataset in the following format:
```json
[
    {
        "time_step": 0.0,
        "ego": [
            0.23675120296251237,
            0.1669917312635742,
            -27.425548112520577,
            -31.747064339316278,
            7.962036500317897,
            112.97880651002266,
            18.464763173475692
        ],
        "controls": [
            -31.747064339316278,
            7.962036500317897,
            112.97880651002266,
            18.464763173475692
        ],
        "vehicles": [
            [
                62.143173925653315,
                248.53770338753876,
                -35.40876196161211,
                22.95118342180229,
                -9.105821032479932,
                64.73580130372206,
                20.870194643288393
            ],
            [
                -112.52492309729413,
                -37.48319499509405,
                -28.88091324282289,
                4.553881571962761,
                3.34055560505324,
                97.24848807184651,
                21.343028501104524
            ],
            [
                -192.1136298853651,
                -219.75403228948414,
                -26.153993060803415,
                -5.8796763648934425,
                19.447370213657194,
                149.58411483606568,
                22.92785782718099
            ]
        ]
    },
]
```

Where the the ego key represents the state of the ego vehicle:
- x,y,z (meters)
- roll,pitch, global heading (degrees)
- velocity (meters/second)

Controls:
- roll_cmd (dgs)
- pitch_cmd (dgs)
- psi_cmd (dgs)
- airspeed_cmd (meters/second)

Vehicles:
- A list of each vehicle within the vicinty with its respective states 