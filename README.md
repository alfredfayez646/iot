# iot

	traffic_volume	holiday	temp	rain_1h	snow_1h	clouds_all	weather_main	weather_description	date_time
0	5545	None	288.28	0.0	0.0	40	Clouds	scattered clouds	02-10-2012 09:00
1	4516	None	289.36	0.0	0.0	75	Clouds	broken clouds	02-10-2012 10:00
2	4767	None	289.58	0.0	0.0	90	Clouds	overcast clouds	02-10-2012 11:00
3	5026	None	290.13	0.0	0.0	90	Clouds	overcast clouds	02-10-2012 12:00
4	4918	None	291.14	0.0	0.0	75	Clouds	broken clouds	02-10-2012 13:00

Dataset Description
The dataset contains hourly data on the traffic volume for westbound I-94, a major interstate highway in the US that connects Minneapolis and St Paul, Minnesota. The data was collected by the Minnesota Department of Transportation (MnDOT) from 2012 to 2018 at a station roughly midway between the two cities.

The dataset has 48204 instances and 9 attributes. The attributes are:

holiday: a categorical variable that indicates whether the date is a US national holiday or a regional holiday (such as the Minnesota State Fair).
temp: a numeric variable that shows the average temperature in kelvin.
rain_1h: a numeric variable that shows the amount of rain in mm that occurred in the hour.
snow_1h: a numeric variable that shows the amount of snow in mm that occurred in the hour.
clouds_all: a numeric variable that shows the percentage of cloud cover.
weather_main: a categorical variable that gives a short textual description of the current weather (such as Clear, Clouds, Rain, etc.).
weather_description: a categorical variable that gives a longer textual description of the current weather (such as light rain, overcast clouds, etc.).
date_time: a datetime variable that shows the hour of the data collected in local CST time.
traffic_volume: a numeric variable that shows the hourly I-94 reported westbound traffic volume.

The dataset has 48204 rows and 9 columns
traffic_volume           int64
holiday                 object
temp                   float64
rain_1h                float64
snow_1h                float64
clouds_all               int64
weather_main            object
weather_description     object
date_time               object
dtype: object

Missing values in each column:
traffic_volume         0
holiday                0
temp                   0
rain_1h                0
snow_1h                0
clouds_all             0
weather_main           0
weather_description    0
date_time              0
dtype: int64

	traffic_volume	temp	rain_1h	snow_1h	clouds_all
count	48204.000000	48204.000000	48204.000000	48204.000000	48204.000000
mean	3259.818355	281.205870	0.334264	0.000222	49.362231
std	1986.860670	13.338232	44.789133	0.008168	39.015750
min	0.000000	0.000000	0.000000	0.000000	0.000000
25%	1193.000000	272.160000	0.000000	0.000000	1.000000
50%	3380.000000	282.450000	0.000000	0.000000	64.000000
75%	4933.000000	291.806000	0.000000	0.000000	90.000000
max	7280.000000	310.070000	9831.300000	0.510000	100.000000

Model Evaluation¶
In this step, we will evaluate the model on the testing data using appropriate metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE). We will also analyze the residuals to check if they follow a normal distribution.

Mean Absolute Error (MAE): 1599.8217805960567
Mean Squared Error (MSE): 3313847.8570689284
Root Mean Squared Error (RMSE): 1820.3977194747658

Best parameters: {'fit_intercept': False}
Best score: -8529192.644074481

result predictions
[2432.11365715 3787.89459413 2840.63610964 4189.35317591 2593.2242563
 3910.51693271 3733.5163106  2815.68312211 3469.91383598 3970.33923232]
