# install library

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf



path  = os.getcwd()
print(path)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# read csv files

## Total charging station in London
total_df = pd.read_csv('total_charging.csv')
total_df.head()
total_df.loc[total_df['Jan-23 \n(Total Charging Devices) [Note 2]']=='x','Jan-23 \n(Total Charging Devices) [Note 2]'] = 0
total_df['Jan-23 \n(Total Charging Devices) [Note 2]'] = total_df['Jan-23 \n(Total Charging Devices) [Note 2]'].str.replace(',','')
total_df['Jan-23 \n(Total Charging Devices) [Note 2]'] = total_df['Jan-23 \n(Total Charging Devices) [Note 2]'].fillna(0).astype(int)
total_df['Jan-23 \n(Total Charging Devices) [Note 2]']
total_df.loc[total_df['Oct-22\n(Total Charging Devices)']=='x','Oct-22\n(Total Charging Devices)'] = 0
total_df['Oct-22\n(Total Charging Devices)'] = total_df['Oct-22\n(Total Charging Devices)'].str.replace(',','')
total_df['Oct-22\n(Total Charging Devices)'] = total_df['Oct-22\n(Total Charging Devices)'].fillna(0).astype(int)
## London total charging station
london_total_df = total_df.loc[220:255]
london_total_df
total_df[total_df['Local Authority / Region Name']=='Inner London']
total_df[total_df['Local Authority / Region Name']=='Outer London'] # 220 is london / 236 is inner london / 255 
## inner london total charging station
il_total_df = total_df[220:236]
il_total_df.info()
il_total_df
## outer london total charging station
ol_tc_df = total_df[236:256]
ol_tc_df.tail()
## Rapid charging station in London
rapid_df = pd.read_csv('rapid_charging.csv')
rapid_df.tail()
london_rapid = rapid_df.loc[220:255]
london_rapid.tail(15)
il_rapid_df = rapid_df[220:236]
il_rapid_df.head()
il_rapid_df['Jan-23\n(Rapid charging or above devices) [Note 4]'] = il_rapid_df['Jan-23\n(Rapid charging or above devices) [Note 4]'].str.replace(',','').astype(int)
il_rapid_df['Oct-22\n(Rapid charging or above devices)'] = il_rapid_df['Oct-22\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
il_rapid_df
ol_rapid_df = rapid_df[236:256]
ol_rapid_df['Jan-23\n(Rapid charging or above devices) [Note 4]'] = ol_rapid_df['Jan-23\n(Rapid charging or above devices) [Note 4]'].str.replace(',','').astype(int)
ol_rapid_df['Oct-22\n(Rapid charging or above devices)'] = ol_rapid_df['Oct-22\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
ol_rapid_df.info()
ol_rapid_df.head()

## ulevs
ulevs = pd.read_csv('ulevs.csv')
ulevs['Fuel'].value_counts()

## company_ulevs_Battery electric
company_ulevs = ulevs.loc[227:262]
company_ulevs['2022 Q3'] = company_ulevs['2022 Q3'].str.replace(',','').astype(int)

## company_ulevs inner London
il_company_ulevs = ulevs.loc[227:242]
il_company_ulevs['2022 Q3'] = il_company_ulevs['2022 Q3'].str.replace(',','').astype(int)
il_company_ulevs.tail()
## company_ulevs outer London
ol_company_ulevs = ulevs.loc[243:262]
ol_company_ulevs['2022 Q3'] = ol_company_ulevs['2022 Q3'].str.replace(',','').astype(int)
ol_company_ulevs

ulevs['Fuel'].value_counts()

# Plug-in hybrid electric (petrol) - company
ulevs[(ulevs['Fuel']== 'Plug-in hybrid electric (petrol)') & (ulevs['ONS Geography [note 6]']=='London')]
company_ulevs_petrol = ulevs.loc[4184:4219]
company_ulevs_petrol['2022 Q3'] = company_ulevs_petrol['2022 Q3'].str.replace(',','').astype(int)
company_ulevs_petrol.head()
## company_ulevs inner London (Plug-in hybrid electric (petrol))
il_company_ulevs_petrol = ulevs.loc[4184:4199]
il_company_ulevs_petrol['2022 Q3'] = il_company_ulevs_petrol['2022 Q3'].str.replace(',','').astype(int)
il_company_ulevs_petrol
## company_ulevs outer London (Plug-in hybrid electric (petrol))
ol_company_ulevs_petrol = ulevs.loc[4200:4219]
ol_company_ulevs_petrol['2022 Q3'] = ol_company_ulevs_petrol['2022 Q3'].str.replace(',','').astype(int)
ol_company_ulevs_petrol

# Plug-in hybrid electric (diesel) - company
ulevs[(ulevs['Fuel']== 'Plug-in hybrid electric (diesel)') & (ulevs['ONS Geography [note 6]']=='London')]
company_ulevs_diesel = ulevs.loc[2884:2919]
company_ulevs_diesel
company_ulevs_diesel.loc[company_ulevs_diesel['2022 Q3']=='[c]','2022 Q3']=0
company_ulevs_diesel['2022 Q3'] = company_ulevs_diesel['2022 Q3'].astype(int)
## company_ulevs inner London (Plug-in hybrid electric (diesel))
il_company_ulevs_diesel = ulevs.loc[2884:2899]
il_company_ulevs_diesel['2022 Q3'] = il_company_ulevs_diesel['2022 Q3'].astype(int)
il_company_ulevs_diesel.info()
## company_ulevs outer London (Plug-in hybrid electric (diesel))
ol_company_ulevs_diesel = ulevs.loc[2900:2919]
ol_company_ulevs_diesel['2022 Q3'] = ol_company_ulevs_diesel['2022 Q3'].astype(int)
ol_company_ulevs_diesel
ol_company_ulevs_diesel.info()

## private_ulevs_Battery electric
ulevs.loc[684:719]
ulevs[(ulevs['ONS Geography [note 6]']=='London') & (ulevs['Keepership']=='Private')]
private_ulevs = ulevs.loc[684:719]
private_ulevs['2022 Q3'] = private_ulevs['2022 Q3'].str.replace(',','').astype(int)
private_ulevs
private_ulevs = private_ulevs.rename(columns={'ONS Geography [note 6]':'Local Authority / Region Name',\
                                              'ONS Code [note 6]':'Local Authority / Region Code'})
private_ulevs.describe()
## private_ulevs inner London
il_private_ulevs = ulevs.loc[684:699]
il_private_ulevs['2022 Q3'] = il_private_ulevs['2022 Q3'].str.replace(',','').astype(int)
il_private_ulevs.info()
## private_ulevs outer London
ol_private_ulevs = ulevs.loc[700:719]
ol_private_ulevs['2022 Q3'] = ol_private_ulevs['2022 Q3'].str.replace(',','').astype(int)
ol_private_ulevs
ol_private_ulevs['ONS Geography [note 6]']


# Plug-in hybrid electric (diesel) - Private
ulevs[(ulevs['Fuel']== 'Plug-in hybrid electric (diesel)') & (ulevs['ONS Geography [note 6]']=='London')]
private_ulevs_diesel = ulevs.loc[3315:3350]
private_ulevs_diesel['2022 Q3'] = private_ulevs_diesel['2022 Q3'].astype(int)
private_ulevs_diesel = private_ulevs_diesel.rename(columns={'ONS Code [note 6]':'Local Authority / Region Code'})
private_ulevs_diesel.head()
## company_ulevs inner London (Plug-in hybrid electric (diesel))
il_company_ulevs_diesel = ulevs.loc[3315:3330]
il_company_ulevs_diesel['2022 Q3'] = il_company_ulevs_diesel['2022 Q3'].astype(int)
il_company_ulevs_diesel
## company_ulevs outer London (Plug-in hybrid electric (diesel))
ol_company_ulevs_diesel = ulevs.loc[3331:3350]
ol_company_ulevs_diesel['2022 Q3'] = ol_company_ulevs_diesel['2022 Q3'].astype(int)
ol_company_ulevs_diesel

# Plug-in hybrid electric (petrol) - Private
ulevs[(ulevs['Fuel']== 'Plug-in hybrid electric (petrol)') & (ulevs['ONS Geography [note 6]']=='London')]
company_ulevs_petrol = ulevs.loc[4634:4669]
company_ulevs_petrol['2022 Q3'] = company_ulevs_petrol['2022 Q3'].str.replace(',','').astype(int)
company_ulevs_petrol
## company_ulevs inner London (Plug-in hybrid electric (diesel))
il_company_ulevs_petrol = ulevs.loc[4634:4649]
il_company_ulevs_petrol['2022 Q3'] = il_company_ulevs_petrol['2022 Q3'].str.replace(',','').astype(int)
il_company_ulevs_petrol
## company_ulevs outer London (Plug-in hybrid electric (diesel))
ol_company_ulevs_petrol = ulevs.loc[4650:4669]
ol_company_ulevs_petrol['2022 Q3'] = ol_company_ulevs_petrol['2022 Q3'].str.replace(',','').astype(int)
ol_company_ulevs_petrol

private_ulevs.head()
london_total_df.head()
# merge with total charging, rapid charging and private_ulevs(batteray) 
total_battery = london_total_df.merge(private_ulevs, on ='Local Authority / Region Code',how = 'outer')
total_rapid_battery = total_battery.merge(london_rapid, on = 'Local Authority / Region Code',how = 'outer')
total_rapid_battery =total_rapid_battery.drop(columns={'2019 Q2','2019 Q1','2018 Q4', '2018 Q3',\
                          '2018 Q2', '2018 Q1', '2017 Q4', '2017 Q3', '2017 Q2', '2017 Q1',\
                           '2016 Q4','2016 Q3','2016 Q2','2016 Q1','2015 Q4',\
                              '2015 Q3','2015 Q2','2015 Q1','2014 Q4',\
                                '2014 Q3','2014 Q2','2014 Q1','2013 Q4','2013 Q3','2013 Q2',\
                                    '2013 Q1','2012 Q4', '2012 Q3',\
                                        '2012 Q2' ,'2012 Q1' ,'2011 Q4'})
total_rapid_battery.keys()
## object to int - EV
total_rapid_battery['2022 Q2'] = total_rapid_battery['2022 Q2'].str.replace(',','').astype(int)
total_rapid_battery['2022 Q1'] = total_rapid_battery['2022 Q1'].str.replace(',','').astype(int)
total_rapid_battery['2021 Q4'] = total_rapid_battery['2021 Q4'].str.replace(',','').astype(int)
total_rapid_battery['2021 Q3'] = total_rapid_battery['2021 Q3'].str.replace(',','').astype(int)
total_rapid_battery['2021 Q2'] = total_rapid_battery['2021 Q2'].str.replace(',','').astype(int)
total_rapid_battery['2021 Q1'] = total_rapid_battery['2021 Q1'].str.replace(',','').astype(int)
total_rapid_battery['2020 Q4'] = total_rapid_battery['2020 Q4'].str.replace(',','').astype(int)
total_rapid_battery['2020 Q3'] = total_rapid_battery['2020 Q3'].str.replace(',','').astype(int)
total_rapid_battery['2020 Q2'] = total_rapid_battery['2020 Q2'].str.replace(',','').astype(int)
total_rapid_battery['2020 Q1'] = total_rapid_battery['2020 Q1'].str.replace(',','').astype(int)
total_rapid_battery['2019 Q4'] = total_rapid_battery['2019 Q4'].str.replace(',','').astype(int)
total_rapid_battery['2019 Q3'] = total_rapid_battery['2019 Q3'].str.replace(',','').astype(int)

## object to int - total station
total_rapid_battery['July-22 \n(Total Charging Devices)'] = total_rapid_battery['July-22 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Apr-22 \n(Total Charging Devices)'] = total_rapid_battery['Apr-22 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Jan-22 \n(Total Charging Devices)2'] = total_rapid_battery['Jan-22 \n(Total Charging Devices)2'].str.replace(',','').astype(int)
total_rapid_battery['Oct-21 \n(Total Charging Devices)'] = total_rapid_battery['Oct-21 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['July-21 \n(Total Charging Devices)'] = total_rapid_battery['July-21 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Apr-21 \n(Total Charging Devices)'] = total_rapid_battery['Apr-21 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Jan-21 \n(Total Charging Devices)'] = total_rapid_battery['Jan-21 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Oct-20 \n(Total Charging Devices)'] = total_rapid_battery['Oct-20 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['July-20 \n(Total Charging Devices)'] = total_rapid_battery['July-20 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Apr-20 \n(Total Charging Devices)'] = total_rapid_battery['Apr-20 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Jan-20 \n(Total Charging Devices)'] = total_rapid_battery['Jan-20 \n(Total Charging Devices)'].str.replace(',','').astype(int)
total_rapid_battery['Oct-19 \n(Total Charging Devices)'] = total_rapid_battery['Oct-19 \n(Total Charging Devices)'].str.replace(',','').astype(int)

## object to int - rapid station
total_rapid_battery['July-22\n(Rapid charging or above devices)'] = total_rapid_battery['July-22\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Apr-22\n(Rapid charging or above devices)'] = total_rapid_battery['Apr-22\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Jan-22\n(Rapid charging or above devices)2'] = total_rapid_battery['Jan-22\n(Rapid charging or above devices)2'].str.replace(',','').astype(int)
total_rapid_battery['Oct-21\n(Rapid charging or above devices)'] = total_rapid_battery['Oct-21\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['July-21\n(Rapid charging or above devices)'] = total_rapid_battery['July-21\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Apr-21\n(Rapid charging or above devices)'] = total_rapid_battery['Apr-21\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Jan-21\n(Rapid charging or above devices)'] = total_rapid_battery['Jan-21\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Oct-20\n(Rapid charging or above devices)'] = total_rapid_battery['Oct-20\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Jul-20\n(Rapid charging or above devices)'] = total_rapid_battery['Jul-20\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Apr-20\n(Rapid charging or above devices)'] = total_rapid_battery['Apr-20\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Jan-20\n(Rapid charging or above devices)'] = total_rapid_battery['Jan-20\n(Rapid charging or above devices)'].str.replace(',','').astype(int)
total_rapid_battery['Oct-19\n(Rapid charging or above devices)'] = total_rapid_battery['Oct-19\n(Rapid charging or above devices)'].str.replace(',','').astype(int)

total_rapid_battery.describe()
total_rapid_battery['Oct-22\n(Rapid charging or above devices)'] = total_rapid_battery['Oct-22\n(Rapid charging or above devices)'].fillna(0).astype(int)
total_rapid_battery.describe()

total_rapid_battery.head()


# see the relationship 
g = sns.pairplot(total_rapid_battery[['Oct-22\n(Total Charging Devices)','2022 Q3','Oct-22\n(Rapid charging or above devices)']])
g.map_upper(sns.regplot)
g.map_lower(sns.regplot)
plt.show()
# regression
# total_rapid_battery_data = total_rapid_battery[['Oct-22\n(Total Charging Devices)','Oct-22\n(Rapid charging or above devices)','2022 Q3',\
#                                                 'Local Authority / Region Code','Local Authority / Region Name']]
total_rapid_battery.info()
total_rapid_battery.head()
total_rapid_battery_data = total_rapid_battery.rename(columns={'2022 Q3':'Private EV population in 2022 Q3'})
total_rapid_battery_data.head()

# total_rapid_battery_data_1 = total_rapid_battery[['Oct-22\n(Total Charging Devices)','Oct-22\n(Rapid charging or above devices)','2022 Q3']]
# total_rapid_battery_data_1
# tf.compat.v1.disable_eager_execution()
# xy = np.array(total_rapid_battery_data_1,dtype=np.float32)
# print(xy)
# x_data = xy[:,0:-1]
# x_data
# y_data = xy[:,[-1]]
# y_data
# X = tf.compat.v1.placeholder(tf.float32, [None, 2])
# Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
# W = tf.Variable(tf.compat.v1.random_normal([2,1]), name='weight')
# b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')
# hypothesis = tf.compat.v1.matmul(X, W) + b
# cost = tf.compat.v1.reduce_mean(tf.square(hypothesis - Y))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# for step in range(10001):
#     cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
#     if step % 500 == 0:
#         print('#', step, 'lost cost: ', cost_)
#         print('-Population of EV: ', hypo_[0])

# Correlation between the number of EVs and the number of charging stations 
import numpy as np
total_rapid_battery_data.head()
total_rapid_battery_data['slow_fast'] = total_rapid_battery_data['Oct-22\n(Total Charging Devices)'] - total_rapid_battery_data['Oct-22\n(Rapid charging or above devices)']
total_rapid_battery_data
car_counts = total_rapid_battery_data[['Local Authority / Region Name','Private EV population in 2022 Q3']].to_dict('records')
ev_result = {c['Local Authority / Region Name']: c['Private EV population in 2022 Q3'] for c in car_counts}
ev_result
car_counts_list = list(ev_result.values())

t_charging_counts = total_rapid_battery_data[['Local Authority / Region Name','Oct-22\n(Total Charging Devices)']].to_dict('records')
total_charging_result = {t['Local Authority / Region Name']:t['Oct-22\n(Total Charging Devices)'] for t in t_charging_counts}
total_charging_result
total_charging_list = list(total_charging_result.values())

corr_coef = np.corrcoef(car_counts_list, total_charging_list)[0, 1]
print(f"The Correlation between the number of EVs and the charging stations by each borough: {corr_coef}") 
total_charging_result
# linear regression - total charging station and EV
from sklearn.linear_model import LinearRegression

X = np.array(list(total_charging_result.values())).reshape(-1, 1)
y = np.array(list(ev_result.values()))

model = LinearRegression()
model.fit(X, y)

num_pois = 100
predicted_car_counts = model.predict([[num_pois]])

print(f"If London has {num_pois} charging stations, {predicted_car_counts[0]:.2f} EV are predicted")


total_rapid_battery_data.head()
# add rapid charging point
r_charging_counts = total_rapid_battery_data[['Local Authority / Region Name','Oct-22\n(Rapid charging or above devices)']].to_dict('records')
rapid_charging_result = {r['Local Authority / Region Name']:r['Oct-22\n(Rapid charging or above devices)'] for r in r_charging_counts}
rapid_charging_result
rapid_charging_result_list = list(rapid_charging_result.values())

for region in ev_result:
    total_charging_result[region] = total_charging_result[region] + rapid_charging_result[region]

print(ev_result)
print(total_charging_result)

# add slow_fast charging point
slow_charging_counts = total_rapid_battery_data[['Local Authority / Region Name','slow_fast']].to_dict('records')
slow_charging_counts = {s['Local Authority / Region Name']:s['slow_fast'] for s in slow_charging_counts}
slow_charging_counts
slow_charging_counts_list = list(slow_charging_counts.values())
slow_charging_counts_list

for region in ev_result:
    total_charging_result[region] = total_charging_result[region] + rapid_charging_result[region] + slow_charging_counts[region]

print(ev_result)
print(slow_charging_counts)


from sklearn.linear_model import LinearRegression

X = np.array(list(zip(total_charging_result.values(), rapid_charging_result.values(), slow_charging_counts.values())))
y = np.array(list(ev_result.values()))

model = LinearRegression()
model.fit(X, y)

num_pois = 100
num_fast_chargers = 40
num_slow_chargers = 60
predicted_car_counts = model.predict([[num_pois, num_fast_chargers, num_slow_chargers]])

print(f"When the number of total charging staion is {num_pois} and the number of rapid charging staion is {num_fast_chargers}\
      and the number of slow_fast charging station is {num_slow_chargers}, predicted number of EVs : {predicted_car_counts[0]:.2f}")

# create scatter plot for total charging stations vs. EV counts
plt.scatter(total_charging_list, car_counts_list)
plt.xlabel('Total charging staions')
plt.ylabel('EV counts')
plt.title('Correlation between Total charging stations and EV counts')

# add regression line
sns.regplot(x=total_charging_list, y=car_counts_list)

plt.show()

# create scatter plot for rapid charging stations vs. EV counts
plt.scatter(rapid_charging_result_list, car_counts_list)
plt.xlabel('Rapid or above charging staions')
plt.ylabel('EV counts')
plt.title('Correlation between Rapid or above charging stations and EV counts')

# add regression line
sns.regplot(x=rapid_charging_result_list, y=car_counts_list)

plt.show()




from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print(total_rapid_battery_data.columns)
total_rapid_battery_data = total_rapid_battery_data[['Local Authority / Region Code',\
                                                     'Local Authority / Region Name',\
                                                     'Oct-22\n(Total Charging Devices)',\
                                                     'Oct-22\n(Rapid charging or above devices)',\
                                                     'slow_fast','Private EV population in 2022 Q3']]
total_rapid_battery_data.head()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 데이터 불러오기

total_rapid_battery_data
# Total Charging Devices와 Rapid Charging Devices 간의 상관관계 분석
corr = total_rapid_battery_data[['Oct-22\n(Total Charging Devices)','Oct-22\n(Rapid charging or above devices)','Private EV population in 2022 Q3']].corr()

# 상관관계 히트맵 그리기
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
total_rapid_battery_data.head()
# 독립변수와 종속변수 지정
X = total_rapid_battery_data[['Oct-22\n(Total Charging Devices)', 'Oct-22\n(Rapid charging or above devices)']]
y = total_rapid_battery_data['Private EV population in 2022 Q3']

# train set과 test set으로 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 예측
y_pred = model.predict(X_test)

# 모델 평가
score = model.score(X_test, y_test)
print('Model Score:', score)

#££££££££££££ inner london
total_rapid_battery_data_inner = total_rapid_battery_data.loc[2:15]
total_rapid_battery_data_inner.keys()
car_counts_inner = total_rapid_battery_data_inner[['Local Authority / Region Name_x','Private EV population in 2022 Q3']].to_dict('records')
ev_result_inner = {c['Local Authority / Region Name_x']: c['Private EV population in 2022 Q3'] for c in car_counts_inner}
ev_result_inner
car_counts_list_inner = list(ev_result_inner.values())

t_charging_counts_inner = total_rapid_battery_data_inner[['Local Authority / Region Name','Oct-22\n(Total Charging Devices)']].to_dict('records')
total_charging_result_inner = {t['Local Authority / Region Name']:t['Oct-22\n(Total Charging Devices)'] for t in t_charging_counts_inner}
total_charging_result_inner
total_charging_list_inner = list(total_charging_result_inner.values())

corr_coef_inner = np.corrcoef(car_counts_list_inner, total_charging_list_inner)[0, 1]
print(f"지역별 자동차 갯수와 주요소 갯수의 상관관계: {corr_coef_inner}") 


X = np.array(list(total_charging_result_inner.values())).reshape(-1, 1)
y = np.array(list(ev_result_inner.values()))

model_inner = LinearRegression()
model_inner.fit(X, y)

num_pois_inner = 100
predicted_car_counts_inner = model_inner.predict([[num_pois_inner]])

print(f"When the number of charging station is {num_pois_inner}, predicted only EV cars : {predicted_car_counts_inner[0]:.2f}")

# add rapid charging point
r_charging_counts_inner = total_rapid_battery_data_inner[['Local Authority / Region Name','Oct-22\n(Rapid charging or above devices)']].to_dict('records')
rapid_charging_result_inner = {r['Local Authority / Region Name']:r['Oct-22\n(Rapid charging or above devices)'] for r in r_charging_counts_inner}
rapid_charging_result_inner
rapid_charging_result_list_inner = list(rapid_charging_result_inner.values())

for region in ev_result_inner:
    total_charging_result_inner[region] = total_charging_result_inner[region] + rapid_charging_result_inner[region]

print(ev_result_inner)
print(total_charging_result_inner)

# add slow_fast charging point
slow_charging_counts_inner = total_rapid_battery_data_inner[['Local Authority / Region Name','slow_fast']].to_dict('records')
slow_charging_counts_inner = {s['Local Authority / Region Name']:s['slow_fast'] for s in slow_charging_counts_inner}
slow_charging_counts_inner
slow_charging_counts_list_inner = list(slow_charging_counts_inner.values())
slow_charging_counts_list_inner

for region in ev_result_inner:
    total_charging_result_inner[region] = total_charging_result_inner[region] + rapid_charging_result_inner[region] + slow_charging_counts_inner[region]

print(ev_result_inner)
print(slow_charging_counts_inner)


from sklearn.linear_model import LinearRegression

X = np.array(list(zip(total_charging_result_inner.values(), rapid_charging_result_inner.values(), slow_charging_counts_inner.values())))
y = np.array(list(ev_result_inner.values()))

model_inner = LinearRegression()
model_inner.fit(X, y)

num_pois_inner = 100
num_fast_chargers_inner = 40
num_slow_chargers_inner = 60
predicted_car_counts_inner = model_inner.predict([[num_pois_inner, num_fast_chargers_inner, num_slow_chargers_inner]])

print(f"When the number of total charging staion is {num_pois_inner} and the number of rapid charging staion is {num_fast_chargers_inner}\
      and the number of slow_fast charging station is {num_slow_chargers_inner}, predicted number of EVs : {predicted_car_counts_inner[0]:.2f}")


# 지역별 charging station per 100,000 population 값 가져오기
total_rapid_battery.head()
total_rapid_battery['Local Authority / Region Name_x']
total_rapid_battery_inner_london = total_rapid_battery.loc[2:15]
total_rapid_battery_inner_london.keys()

total_rapid_battery = total_rapid_battery.rename(columns = {total_rapid_battery.columns[5]:'total per 100,000 population'})
total_rapid_battery.head()
total_rapid_battery_oct = total_rapid_battery[['Local Authority / Region Name',\
                                                                        'Local Authority / Region Code',\
                                                                        '2022 Q3',\
                                                                        'Oct-22\n(Total Charging Devices)',\
                                                                        'total per 100,000 population',\
                                                                        'Oct-22\n(Rapid charging or above devices)',\
                                                                        'Oct-22\n(per 100,000 population)']]
total_rapid_battery_oct
total_rapid_battery_oct['Oct-22\n(per 100,000 population)'] = total_rapid_battery_oct['Oct-22\n(per 100,000 population)'].astype(float)
total_rapid_battery_oct = total_rapid_battery_oct.rename(columns = {total_rapid_battery_oct.columns[-1]:'rapid per 100,000 population'})
total_rapid_battery_oct


station_density = total_rapid_battery_oct['total per 100,000 population']

# check the total charging station density inner London
plt.bar(total_rapid_battery_oct['Local Authority / Region Name'], station_density)
plt.xticks(rotation=90)
plt.xlabel('location')
plt.ylabel('density per 10,000')
plt.title('Total location density')
plt.show()

# check the rapid charging station density inner London
rapid_station_density = total_rapid_battery_oct['rapid per 100,000 population']
plt.bar(total_rapid_battery_oct['Local Authority / Region Name'], rapid_station_density)
plt.xticks(rotation=90)
plt.xlabel('location')
plt.ylabel('density per 10,000')
plt.title('Rapid location density')
plt.show()

total_rapid_battery_oct.head()
# heatmap - correlation Total charging devices and the population inner London
corr_lon = total_rapid_battery_oct[['Oct-22\n(Total Charging Devices)','2022 Q3','Oct-22\n(Rapid charging or above devices)']].corr()

# 상관관계 히트맵 그리기
x_axis_labels = ['Charging station','Population of EV','Rapid charging ']
y_axis_labels = ['Charging station','Population of EV','Rapid charging ']
sns.heatmap(corr_lon, annot=True, cmap='coolwarm', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title('Correlation Heatmap')
plt.show()


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import hstack

# 데이터 불러오기

total_rapid_battery_oct['slow_fast'] = total_rapid_battery_oct['Oct-22\n(Total Charging Devices)']-total_rapid_battery_oct['Oct-22\n(Rapid charging or above devices)']
total_rapid_battery_oct
# 필요한 모듈을 import 합니다
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# 데이터를 불러옵니다
total_rapid_battery_oct
# 필요한 feature들을 추출합니다
df = total_rapid_battery_oct[['Local Authority / Region Name', 'Oct-22\n(Total Charging Devices)',
         'Oct-22\n(Rapid charging or above devices)','2022 Q3']]

# One-hot encoding을 수행합니다
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = ct.fit_transform(df.iloc[:, :-1])

y = df.iloc[:, -1].values

# Linear Regression 모델을 정의하고 학습합니다
model = LinearRegression()
model.fit(X, y)

# 새로운 데이터를 예측합니다
new_data = pd.DataFrame({'Local Authority / Region Name': ['Newham'],
                         'Oct-22\n(Total Charging Devices)': [700],
                         'Oct-22\n(Rapid charging or above devices)': [100],
                         'slow_fast':[500]})
new_data_transformed = ct.transform(new_data)
prediction = model.predict(new_data_transformed)

# 결과를 출력합니다
print(f"Predicted population of EVs : {prediction[0]:.2f}.")

#########################################################
# extract the boroughs whose rapid charging station is over 24 - the average is 24 
total_rapid_battery_oct.describe()
df = total_rapid_battery_oct[total_rapid_battery_oct['Oct-22\n(Rapid charging or above devices)'] >= 24]
df.head()
# X, y data define
X = df[['Local Authority / Region Code', 'Oct-22\n(Total Charging Devices)', 'Oct-22\n(Rapid charging or above devices)']]
y = df['2022 Q3']

# slpit test data and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# ColumnTransformer 
ct = ColumnTransformer([('encoder', OneHotEncoder(), ['Local Authority / Region Code'])], remainder='passthrough')
X_transformed = ct.fit_transform(X)
# X_train preprocessing
X_train_transformed = ct.transform(X_train)

# LinearRegression
model = LinearRegression()

# train data trains LinearRegression
model.fit(X_train_transformed, y_train)

# X_test preprocessing
X_test_transformed = ct.transform(X_test)


# predict LinearRegression using text 
y_pred = model.predict(X_test_transformed)|

for i in range(len(X_test)):
    predicted_ev = y_pred[i]
    print(f"{X_test.iloc[i]['Local Authority / Region Code']}: predicted number of EVs is {predicted_ev:.2f}")



# evaluate the model
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 score: ", r2)

# predicted model's value
from sklearn.linear_model import LinearRegression

y_pred = model.predict(X_test_transformed)
y_pred
# the difference between the predicted and actual values
residuals = y_test - y_pred
residuals
# visualizes the residuals 
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()


# Predicting the number of EVs based on the number of rapid charging stations



### the ratio of rapid charging stations
ev_result
total_charging_result
rapid_charging_result

for city in ev_result.keys():
    fast_charging_rate = rapid_charging_result[city] / total_charging_result[city] *100
    print(f"{city}: {fast_charging_rate:.1f}%")


###
from sklearn.linear_model import LinearRegression

#the number of EV, rapid charging and total charging 
ev_result
total_charging_result
del total_charging_result['London']
del total_charging_result['Inner London']
del total_charging_result['Outer London']
total_charging_result
rapid_charging_result
del rapid_charging_result['London']
del rapid_charging_result['Inner London']
del rapid_charging_result['Outer London']

total_charging_list = list(total_charging_result.values())
rapid_charging_result_list = list(rapid_charging_result.values())
boro = list(total_charging_result.keys())
boro

len(car_counts_list)
len(total_charging_list)
total_charging_list
rapid_charging_result_list

# the ratio of rapid charging station - 30%  
limited_fast_ratio = 0.3

# correlation between EV and rapid charging 
X = [[ratio] for ratio in [fc/st for st, fc in zip(total_charging_list, rapid_charging_result_list)]]
y = car_counts_list

model = LinearRegression()
model.fit(X, y)

# When the rapid is a certain percent, predict EV 
predicted_electric_cars = []
for i in range(len(car_counts_list)):
    # current rapid ratio 
    total_stations = total_charging_list[i]
    fast_ratio = rapid_charging_result_list[i] / total_stations
    
    # predict ratio 
    limited_fast_charging_stations = int(total_stations * limited_fast_ratio)
    
    # total - predicted ratio
    remaining_stations = total_stations - limited_fast_charging_stations
    
    # apply for the predicted ratio, predict the EV 
    predicted_ratio = model.predict([[remaining_stations / total_stations]])
    predicted_electric_cars.append(int(predicted_ratio[0] * remaining_stations))

predicted_electric_cars
result_dict = dict(zip(boro,predicted_electric_cars))
result_dict
# result
for key,value in result_dict.items():
    print(f"{key}: predicted EVs are {value}")



fig = np.arange(len(result_dict))
ax = plt.subplot(111)
ax.bar(fig, result_dict.values(), width=0.2, color='pink', align='center')
ax.bar(fig-1, ev_result.values(), width=0.2, color='g', align='center')
ax.legend(('Predicted','Current'))
plt.xticks(fig,ev_result.keys() )
plt.xticks(rotation=0)
plt.xticks(ha='center', rotation=-90)
plt.ylabel('EV')
plt.title('Forecast EVs - Rapid charging station increase by 30% - linear regression')
# 그래프 출력
plt.show()


# Set the new ratio of rapid charging stations to 40%
rapid_charging_ratio = 0.3

# Calculate the number of rapid charging stations and regular charging stations
total_charging_list  # list of total charging stations at each location
rapid_charging_stations = [int(total_stations * rapid_charging_ratio) for total_stations in total_charging_list]
regular_charging_stations = [total_stations - rapid_charging_stations[i] for i, total_stations in enumerate(total_charging_list)]

# Create the input matrix X and output vector y for linear regression
X = [[rc/total_stations, rc/(rc + rc_reg)] for rc, total_stations, rc_reg in zip(rapid_charging_stations, total_charging_list, regular_charging_stations)]
y = car_counts_list # list of number of electric cars at each location

# Fit a linear regression model to the data
model_r = LinearRegression()
model_r.fit(X, y)

# When a certain ratio of rapid charging stations is desired, predict the number of electric cars
predicted_ev = []
for i in range(len(total_charging_list)):
    # Calculate the number of rapid charging stations and regular charging stations based on the desired ratio
    desired_rapid_charging_stations = int(total_charging_list[i] * rapid_charging_ratio)
    desired_regular_charging_stations = total_charging_list[i] - desired_rapid_charging_stations
    
    # Predict the number of electric cars based on the desired ratio
    predicted_ratio = model_r.predict([[desired_rapid_charging_stations/total_charging_list[i], desired_rapid_charging_stations/(desired_rapid_charging_stations +  regular_charging_stations[i])]])
    predicted_ev.append(int(predicted_ratio[0] * total_charging_list[i]))

predicted_ev




###############
total_charging_result
ev_result_list = list(ev_result.values())
ev_result
# correlation between EV and rapid charging 
X = np.array(list(total_charging_result.values())).reshape(-1, 1)
y = ev_result_list

model = LinearRegression()
model.fit(X, y)

# When the total is increased by 30%, predict EV 
predicted_electric_cars = []
for i in range(len(car_counts_list)):
    # 
    total_stations = total_charging_list[i]
    
    # predict ratio 
    increased_total_stations = int(total_stations)+int(total_stations * 0.3)
    
    predicted_car_counts = model.predict([[increased_total_stations]])
    
    predicted_electric_cars.append(int(predicted_car_counts))

result_dict = dict(zip(boro, predicted_electric_cars))
result_dict_list = list(result_dict.values())
result_dict_list
car_counts_list
ev_result
# result
for key, value in result_dict.items():
    print(f"{key}: predicted EVs are {value}")


fig = np.arange(len(result_dict))
ax = plt.subplot(111)
ax.bar(fig, result_dict.values(), width=0.2, color='pink', align='center')
ax.bar(fig, ev_result.values(), width=0.2, color='g', align='center')
ax.legend(('Predicted','Current'))
plt.xticks(fig,ev_result.keys() )
plt.xticks(rotation=0)
plt.xticks(ha='center', rotation=-90)
plt.ylabel('EV')
plt.title('Forecast EVs - Total charging station increase by 30% - linear regression')
# 그래프 출력
plt.show()




### simple math
# 서울 지역에서 급속충전소 비율을 20%로 제한하여 전기차 수요 예측
limited_fast_charging_rate = 0.2

limited_fast_charging_stations = total_charging_result["Camden"] * limited_fast_charging_rate
usable_charging_stations = total_charging_result["Camden"] - limited_fast_charging_stations

electric_car_demand = usable_charging_stations * fast_charging_rate

print(f"Camden 지역 전기차 수요 예측: {electric_car_demand:.0f} 대")



########
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
total_rapid_battery.keys()
total_rapid_battery_data.keys()
total_rapid_battery_data.head()
total_rapid_battery = total_rapid_battery.drop(16)
total_rapid_battery = total_rapid_battery.drop(0)
total_rapid_battery = total_rapid_battery.drop(1)
total_rapid_battery['Local Authority / Region Name_x']
total_rapid_battery = total_rapid_battery.reset_index(drop=True)
total_rapid_battery.tail()
total_rapid_battery.head()
# Filter the data to keep only necessary columns
total_rapid_battery_data = total_rapid_battery[['Local Authority / Region Code', 'Local Authority / Region Name_x',
        'Oct-22\n(Total Charging Devices)',
       'July-22 \n(Total Charging Devices)',
       'Apr-22 \n(Total Charging Devices)',
       'Jan-22 \n(Total Charging Devices)2',
       'Oct-21 \n(Total Charging Devices)',
       'July-21 \n(Total Charging Devices)',
       'Apr-21 \n(Total Charging Devices)',
       'Jan-21 \n(Total Charging Devices)',
       'Oct-20 \n(Total Charging Devices)',
       'July-20 \n(Total Charging Devices)',
       'Apr-20 \n(Total Charging Devices)',
       'Jan-20 \n(Total Charging Devices)',
       'Oct-19 \n(Total Charging Devices)',
         '2022 Q3',
         '2022 Q2', 
         '2022 Q1', 
         '2021 Q4', 
         '2021 Q3', 
         '2021 Q2', 
         '2021 Q1',
         '2020 Q4', 
         '2020 Q3', 
         '2020 Q2', 
         '2020 Q1', 
         '2019 Q4', 
         '2019 Q3',
         'Oct-22\n(Rapid charging or above devices)',
         'July-22\n(Rapid charging or above devices)',
         'Apr-22\n(Rapid charging or above devices)',
         'Jan-22\n(Rapid charging or above devices)2',
         'Oct-21\n(Rapid charging or above devices)',
         'July-21\n(Rapid charging or above devices)',
         'Apr-21\n(Rapid charging or above devices)',
         'Jan-21\n(Rapid charging or above devices)',
         'Oct-20\n(Rapid charging or above devices)',
         'Jul-20\n(Rapid charging or above devices)',
         'Apr-20\n(Rapid charging or above devices)',
         'Jan-20\n(Rapid charging or above devices)',
         'Oct-19\n(Rapid charging or above devices)']]
total_rapid_battery_data.keys()
total_rapid_battery_data = total_rapid_battery_data.reset_index(drop = True)
total_rapid_battery_data


# Create a new column for restricted number of rapid charging stations
total_rapid_battery_data['Restricted Rapid Charging'] = total_rapid_battery_data['Oct-22\n(Total Charging Devices)'] * 0.1

# Split the data into features and target
X = total_rapid_battery_data.drop(['Local Authority / Region Code', 'Local Authority / Region Name_x', '2022 Q3',
         '2022 Q2', 
         '2022 Q1', 
         '2021 Q4', 
         '2021 Q3', 
         '2021 Q2', 
         '2021 Q1',
         '2020 Q4', 
         '2020 Q3', 
         '2020 Q2', 
         '2020 Q1', 
         '2019 Q4', 
         '2019 Q3'], axis=1)
X = X.reset_index(drop =True)
X.keys()
X
total_rapid_battery_data.head()
car_dic= total_rapid_battery_data[['2022 Q3','2022 Q2','2022 Q1', '2021 Q4', '2021 Q3', '2021 Q2', '2021 Q1','2020 Q4', '2020 Q3', '2020 Q2', '2020 Q1', '2019 Q4', '2019 Q3']]
y = car_dic
combined = pd.concat([car_dic[col] for col in car_dic.columns], ignore_index=True)
combined
# y = combined


# Train a linear regression model
model = LinearRegression().fit(X, y)

# Make predictions for restricted number of rapid charging stations 
X_new = X.copy()
X_new.head()
X_new['Restricted Rapid Charging'] = X_new['Oct-22\n(Total Charging Devices)'] * 0.1
y_pred = model.predict(X_new)

# Calculate predicted number of EVs
pred_num_ev = y_pred.sum()

print(f"Predicted number of EVs with 10% restricted rapid charging stations: {pred_num_ev}")

import pandas as pd
import numpy as np
from fbprophet import Prophet

# Load the data

total_rapid_battery_data.keys()
total_rapid_battery_data
# rename the columns to match Prophet's requirements
import pandas as pd

# Create a list of quarter column headers
quarter_cols = ['Oct-22\n(Rapid charging or above devices)',
       'July-22\n(Rapid charging or above devices)',
       'Apr-22\n(Rapid charging or above devices)',
       'Jan-22\n(Rapid charging or above devices)2',
       'Oct-21\n(Rapid charging or above devices)',
       'July-21\n(Rapid charging or above devices)',
       'Apr-21\n(Rapid charging or above devices)',
       'Jan-21\n(Rapid charging or above devices)',
       'Oct-20\n(Rapid charging or above devices)',
       'Jul-20\n(Rapid charging or above devices)',
       'Apr-20\n(Rapid charging or above devices)',
       'Jan-20\n(Rapid charging or above devices)',
       'Oct-19\n(Rapid charging or above devices)']

total_rapid_battery_data.keys()
total_rapid_battery_data_new = total_rapid_battery_data.drop(columns=['2022 Q3', '2022 Q2', '2022 Q1', '2021 Q4', '2021 Q3', '2021 Q2', '2021 Q1', '2020 Q4', '2020 Q3', '2020 Q2', '2020 Q1', '2019 Q4', '2019 Q3'])
total_rapid_battery_data_new
total_rapid_battery_data_new.tail()
total_rapid_battery_data_new['Local Authority / Region Name_x']
total_rapid_battery_data_new
# Melt the dataframe into a long format
melted = pd.melt(total_rapid_battery_data_new, id_vars=['Local Authority / Region Code', 'Local Authority / Region Name_x','Restricted Rapid Charging'], var_name='Date', value_name='Number of Stations')
melted.keys()

# Rename the date values by extracting the month and year from the column names
melted['Date'] = melted['Date'].apply(lambda x: '-'.join(x.split('\n')[0:2]))
melted['Date']

# Print the result

print(melted)
melted_new = pd.concat([melted,combined], axis=1)
melted_new.to_csv('melted_new.csv')
melted_new.tail(10)
melted_new.keys()
melted_new = melted_new.rename(columns={0:'EV'})
melted_new.tail()


import pandas as pd

# extract month and year from the string and create a valid datetime string
melted_new['Date'] = melted_new['Date'].apply(lambda x: '01-' + x.split('-')[0] + '-' + x.split('-')[1][-2:])
melted_new['Date']
# convert to datetime
melted_new['Date'] = melted_new['Date'].str.replace('July','Jul')
melted_new['Date'] = pd.to_datetime(melted_new['Date'], format='%d-%b-%y')
melted_new['EV'].info()
melted_new.head()
melted_new['Date'].info()
# Group the data by region
grouped_data = melted_new.groupby('Local Authority / Region Code')
grouped_data
# Create an empty DataFrame to store the forecasts
forecasts_by_region = pd.DataFrame()


from fbprophet.diagnostics import cross_validation


for name, group in grouped_data:
    # Rename columns to 'ds' and 'y'
    group = group.rename(columns={'Date': 'ds', 'EV': 'y'})
   
    # Fit the model to the group's data
    model = Prophet()
    model.fit(group)
    
    # Make a dataframe with the future dates
    future = model.make_future_dataframe(periods=365)
    
    # Make the forecasts for the group's data
    forecast = model.predict(future)
    
    # Add a column with the region code to the forecasts
    forecast['region_code'] = name
    
    # Add the forecasts to the DataFrame
    forecasts_by_region = pd.concat([forecasts_by_region, forecast], ignore_index=True)

forecasts_by_region

forecasts_by_region['region_code'].unique()
forecasts_by_region

import matplotlib.pyplot as plt

# choose a region to plot
region = 'E09000014'

# filter the data to the selected region
region_data = forecasts_by_region[forecasts_by_region['region_code'] == region]
print(region_data.tail())
# plot the data
plt.plot(region_data['ds'], region_data['yhat'])

# add title and axis labels
plt.title(f"Forecast for {region}")
plt.xlabel("Date")
plt.ylabel("Forecasted Value")

# show the plot
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns

# 데이터 프레임 불러오기

# 필요한 열만 추출
total_rapid_battery_data.keys()
df = total_rapid_battery_data[['Local Authority / Region Code', 'Local Authority / Region Name_x','2019 Q3', '2019 Q4', '2020 Q1', '2020 Q2', '2020 Q3', '2020 Q4', '2021 Q1', '2021 Q2', '2021 Q3', '2021 Q4', '2022 Q1', '2022 Q2', '2022 Q3',
        'Oct-22\n(Total Charging Devices)',
       'July-22 \n(Total Charging Devices)',
       'Apr-22 \n(Total Charging Devices)',
       'Jan-22 \n(Total Charging Devices)2',
       'Oct-21 \n(Total Charging Devices)',
       'July-21 \n(Total Charging Devices)',
       'Apr-21 \n(Total Charging Devices)',
       'Jan-21 \n(Total Charging Devices)',
       'Oct-20 \n(Total Charging Devices)',
       'July-20 \n(Total Charging Devices)',
       'Apr-20 \n(Total Charging Devices)',
       'Jan-20 \n(Total Charging Devices)',
       'Oct-19 \n(Total Charging Devices)',
       'Oct-22\n(Rapid charging or above devices)',
       'July-22\n(Rapid charging or above devices)',
       'Apr-22\n(Rapid charging or above devices)',
       'Jan-22\n(Rapid charging or above devices)2',
       'Oct-21\n(Rapid charging or above devices)',
       'July-21\n(Rapid charging or above devices)',
       'Apr-21\n(Rapid charging or above devices)',
       'Jan-21\n(Rapid charging or above devices)',
       'Oct-20\n(Rapid charging or above devices)',
       'Jul-20\n(Rapid charging or above devices)',
       'Apr-20\n(Rapid charging or above devices)',
       'Jan-20\n(Rapid charging or above devices)',
       'Oct-19\n(Rapid charging or above devices)']]
df.keys()
# 지역별 전기차 수와 고속 충전소 수를 추출
electric_cars = df[['Local Authority / Region Code','2022 Q3', '2022 Q2', '2022 Q1', '2021 Q4', '2021 Q3', '2021 Q2', '2021 Q1', '2020 Q4', '2020 Q3', '2020 Q2', '2020 Q1', '2019 Q4', '2019 Q3']]
rapid_chargers = df[['Local Authority / Region Code', 'Local Authority / Region Name_x',
                     'Oct-22\n(Total Charging Devices)',
                     'July-22 \n(Total Charging Devices)',
                     'Apr-22 \n(Total Charging Devices)',
                     'Jan-22 \n(Total Charging Devices)2',
                     'Oct-21 \n(Total Charging Devices)',
                     'July-21 \n(Total Charging Devices)',
                     'Apr-21 \n(Total Charging Devices)',
                     'Jan-21 \n(Total Charging Devices)',
                     'Oct-20 \n(Total Charging Devices)',
                     'July-20 \n(Total Charging Devices)',
                     'Apr-20 \n(Total Charging Devices)',
                     'Jan-20 \n(Total Charging Devices)',
                     'Oct-19 \n(Total Charging Devices)',
                     'Oct-22\n(Rapid charging or above devices)',
                     'July-22\n(Rapid charging or above devices)',
                     'Apr-22\n(Rapid charging or above devices)',
                     'Jan-22\n(Rapid charging or above devices)2',
                     'Oct-21\n(Rapid charging or above devices)',
                     'July-21\n(Rapid charging or above devices)',
                     'Apr-21\n(Rapid charging or above devices)',
                     'Jan-21\n(Rapid charging or above devices)',
                     'Oct-20\n(Rapid charging or above devices)',
                     'Jul-20\n(Rapid charging or above devices)',
                     'Apr-20\n(Rapid charging or above devices)',
                     'Jan-20\n(Rapid charging or above devices)',
                     'Oct-19\n(Rapid charging or above devices)']]

# 전기차 수와 고속 충전소 수를 합쳐서 새로운 데이터 프레임 생성
df_merged = pd.merge(electric_cars, rapid_chargers, on='Local Authority / Region Code')
df_merged.head()
df_merged.keys()
# 상관계수 계산
correlation = df_merged.corr()

# Heatmap으로 상관계수 시각화
sns.heatmap(correlation)
plt.show()



import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


# Filter the data to keep only necessary columns
total_rapid_battery_data = total_rapid_battery[['Local Authority / Region Code', 'Local Authority / Region Name_x',
         'Oct-22\n(Rapid charging or above devices)',
         'July-22\n(Rapid charging or above devices)',
         'Apr-22\n(Rapid charging or above devices)',
         'Jan-22\n(Rapid charging or above devices)2',
         'Oct-21\n(Rapid charging or above devices)',
         'July-21\n(Rapid charging or above devices)',
         'Apr-21\n(Rapid charging or above devices)',
         'Jan-21\n(Rapid charging or above devices)',
         'Oct-20\n(Rapid charging or above devices)',
         'Jul-20\n(Rapid charging or above devices)',
         'Apr-20\n(Rapid charging or above devices)',
         'Jan-20\n(Rapid charging or above devices)',
         'Oct-19\n(Rapid charging or above devices)']]

# Create a new column for restricted number of rapid charging stations -30%
total_rapid_battery_data['Restricted Rapid Charging'] = total_rapid_battery_data['Oct-22\n(Rapid charging or above devices)'] * 0.3

# Split the data into features and target
X = total_rapid_battery_data.drop(['Local Authority / Region Code', 'Local Authority / Region Name_x'], axis=1)
y = total_rapid_battery_data[['Oct-22\n(Rapid charging or above devices)']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Ridge(alpha=1.0)

# Perform cross-validation to tune the hyperparameters
param_grid = {'alpha': [0.1, 1, 10]}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and use them to train the model
best_alpha = grid_search.best_params_['alpha']
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train, y_train)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree with max depth 3
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print the predicted number of EVs for each row of the new data
for i, predicted_ev in enumerate(y_pred):
    print(f"Row {i}: predicted number of EVs is {predicted_ev:.2f}")

import matplotlib.pyplot as plt

# Group the data by region and transpose it
region_data = total_rapid_battery_data.groupby('Local Authority / Region Name_x').sum().T

# Plot the data
plt.plot(region_data)
plt.xlabel('Date')
plt.ylabel('Number of rapid charging stations')
plt.legend(region_data.columns)
plt.show()

# Fit a decision tree with varying depths and calculate the accuracy
depths = range(1, 11)
accuracies = []
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the accuracy as a function of depth
plt.scatter(depths, accuracies)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()

total_rapid_battery_data
import matplotlib.pyplot as plt

# Group the data by borough and sum the number of EVs for each quarter
ev_by_borough = total_rapid_battery_data.groupby('Local Authority / Region Name_x')[['Oct-22\n(Rapid charging or above devices)',
                                                                                  'July-22\n(Rapid charging or above devices)',
                                                                                  'Apr-22\n(Rapid charging or above devices)',
                                                                                  'Jan-22\n(Rapid charging or above devices)2',
                                                                                  'Oct-21\n(Rapid charging or above devices)',
                                                                                  'July-21\n(Rapid charging or above devices)',
                                                                                  'Apr-21\n(Rapid charging or above devices)',
                                                                                  'Jan-21\n(Rapid charging or above devices)',
                                                                                  'Oct-20\n(Rapid charging or above devices)',
                                                                                  'Jul-20\n(Rapid charging or above devices)',
                                                                                  'Apr-20\n(Rapid charging or above devices)',
                                                                                  'Jan-20\n(Rapid charging or above devices)',
                                                                                  'Oct-19\n(Rapid charging or above devices)']].sum()

# Plot the number of EVs for each borough
ev_by_borough.plot(kind='bar', figsize=(12, 6))
plt.title('Number of EVs by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of EVs')
plt.show()




# Load libraries
import pandas as pd
from fbprophet import Prophet

total_rapid_battery_data_forecast = total_rapid_battery[['Local Authority / Region Code', 'Local Authority / Region Name_x',
        'Oct-22\n(Total Charging Devices)',
       'July-22 \n(Total Charging Devices)',
       'Apr-22 \n(Total Charging Devices)',
       'Jan-22 \n(Total Charging Devices)2',
       'Oct-21 \n(Total Charging Devices)',
       'July-21 \n(Total Charging Devices)',
       'Apr-21 \n(Total Charging Devices)',
       'Jan-21 \n(Total Charging Devices)',
       'Oct-20 \n(Total Charging Devices)',
       'July-20 \n(Total Charging Devices)',
       'Apr-20 \n(Total Charging Devices)',
       'Jan-20 \n(Total Charging Devices)',
       'Oct-19 \n(Total Charging Devices)',
         '2022 Q3',
         '2022 Q2', 
         '2022 Q1', 
         '2021 Q4', 
         '2021 Q3', 
         '2021 Q2', 
         '2021 Q1',
         '2020 Q4', 
         '2020 Q3', 
         '2020 Q2', 
         '2020 Q1', 
         '2019 Q4', 
         '2019 Q3',
         'Oct-22\n(Rapid charging or above devices)',
         'July-22\n(Rapid charging or above devices)',
         'Apr-22\n(Rapid charging or above devices)',
         'Jan-22\n(Rapid charging or above devices)2',
         'Oct-21\n(Rapid charging or above devices)',
         'July-21\n(Rapid charging or above devices)',
         'Apr-21\n(Rapid charging or above devices)',
         'Jan-21\n(Rapid charging or above devices)',
         'Oct-20\n(Rapid charging or above devices)',
         'Jul-20\n(Rapid charging or above devices)',
         'Apr-20\n(Rapid charging or above devices)',
         'Jan-20\n(Rapid charging or above devices)',
         'Oct-19\n(Rapid charging or above devices)']]

total_rapid_battery_data_forecast.head()
total_rapid_battery_data_forecast.keys()
total_rapid_battery_data_forecast
id_vars = total_rapid_battery_data_forecast[['Local Authority / Region Code', 'Local Authority / Region Name_x']]
id_vars
import re

cols1 = total_rapid_battery_data_forecast[['Oct-22\n(Total Charging Devices)', 'July-22 \n(Total Charging Devices)', 'Apr-22 \n(Total Charging Devices)',
        'Jan-22 \n(Total Charging Devices)2', 'Oct-21 \n(Total Charging Devices)', 'July-21 \n(Total Charging Devices)',
        'Apr-21 \n(Total Charging Devices)', 'Jan-21 \n(Total Charging Devices)', 'Oct-20 \n(Total Charging Devices)',
        'July-20 \n(Total Charging Devices)', 'Apr-20 \n(Total Charging Devices)', 'Jan-20 \n(Total Charging Devices)',
        'Oct-19 \n(Total Charging Devices)']]


cols2 = total_rapid_battery_data_forecast[['Oct-22\n(Rapid charging or above devices)', 'July-22\n(Rapid charging or above devices)',
         'Apr-22\n(Rapid charging or above devices)', 'Jan-22\n(Rapid charging or above devices)2',
         'Oct-21\n(Rapid charging or above devices)', 'July-21\n(Rapid charging or above devices)',
         'Apr-21\n(Rapid charging or above devices)', 'Jan-21\n(Rapid charging or above devices)',
         'Oct-20\n(Rapid charging or above devices)', 'Jul-20\n(Rapid charging or above devices)',
         'Apr-20\n(Rapid charging or above devices)', 'Jan-20\n(Rapid charging or above devices)',
         'Oct-19\n(Rapid charging or above devices)']]

cols1 = cols1.rename(columns={'Oct-22\n(Total Charging Devices)':'2203'})
cols1 = cols1.rename(columns={'July-22 \n(Total Charging Devices)':'2202'})
cols1 = cols1.rename(columns={'Apr-22 \n(Total Charging Devices)':'2201'})
cols1 = cols1.rename(columns={'Jan-22 \n(Total Charging Devices)2':'2104'})
cols1 = cols1.rename(columns={'Oct-21 \n(Total Charging Devices)':'2103'})
cols1 = cols1.rename(columns={'July-21 \n(Total Charging Devices)':'2102'})
cols1 = cols1.rename(columns={'Apr-21 \n(Total Charging Devices)':'2101'})
cols1 = cols1.rename(columns={'Jan-21 \n(Total Charging Devices)':'2004'})
cols1 = cols1.rename(columns={'Oct-20 \n(Total Charging Devices)':'2003'})
cols1 = cols1.rename(columns={'July-20 \n(Total Charging Devices)':'2002'})
cols1 = cols1.rename(columns={'Apr-20 \n(Total Charging Devices)':'2001'})
cols1 = cols1.rename(columns={'Jan-20 \n(Total Charging Devices)':'1904'})
cols1 = cols1.rename(columns={'Oct-19 \n(Total Charging Devices)':'1903'})
cols1


cols2 = cols2.rename(columns={'Oct-22\n(Rapid charging or above devices)':'2203'})
cols2 = cols2.rename(columns={'July-22\n(Rapid charging or above devices)':'2202',
                              'Apr-22\n(Rapid charging or above devices)':'2201',
                              'Jan-22\n(Rapid charging or above devices)2':'2104',
                              'Oct-21\n(Rapid charging or above devices)':'2103',
                              'July-21\n(Rapid charging or above devices)':'2102',
                              'Apr-21\n(Rapid charging or above devices)':'2101',
                              'Jan-21\n(Rapid charging or above devices)':'2004',
                              'Oct-20\n(Rapid charging or above devices)':'2003',
                              'Jul-20\n(Rapid charging or above devices)':'2002',
                              'Apr-20\n(Rapid charging or above devices)':'2001',
                              'Jan-20\n(Rapid charging or above devices)':'1904',
                              'Oct-19\n(Rapid charging or above devices)':'1903'})

cols3 = total_rapid_battery_data_forecast[['2022 Q3', '2022 Q2', '2022 Q1',
       '2021 Q4', '2021 Q3', '2021 Q2', '2021 Q1', '2020 Q4', '2020 Q3',
       '2020 Q2', '2020 Q1', '2019 Q4', '2019 Q3']]
cols3

total_rapid_battery_data_forecast.keys()
year_quarter1 = [re.findall(r'\d{2}', c) for c in cols1]
year_quarter1
year_quarter2 = [re.findall(r'\d{2}', c) for c in cols2]
year_quarter2


new_cols1 = [f"{y}{q}" for y, q in year_quarter1]
new_cols1
new_cols2 = [f"{y}{q}" for y, q in year_quarter2]
new_cols2
total_rapid_battery_data_forecast.head()
df_ev = pd.melt(total_rapid_battery_data_forecast, id_vars=id_vars, value_vars=cols3, var_name="YearQuarter3", value_name="TotalEV")
df_ev
df_ev.keys()
df_ev.drop(columns={'Local Authority / Region Code', 'Local Authority / Region Name_x'},inplace =True)
df_ev
df = pd.melt(total_rapid_battery_data_forecast, id_vars=id_vars, value_vars=cols1, var_name="YearQuarter1", value_name="TotalChargingDevices")
df
df_ra = pd.melt(total_rapid_battery_data_forecast, id_vars=id_vars, value_vars=cols2, var_name="YearQuarter2", value_name="RapidChargingDevices")
df_ra.keys()
df_ra_ = df_ra[['YearQuarter2', 'RapidChargingDevices']]
df_ra_.head()
total_rapid_forecast = pd.concat([df,df_ra_,df_ev], axis=1)
total_rapid_forecast
total_rapid_forecast = total_rapid_forecast.drop(columns={'YearQuarter2','YearQuarter3'})
total_rapid_forecast.head()
total_rapid_forecast['YearQuarter1'].unique()
total_rapid_forecast['YearQuarter1'] = total_rapid_forecast['YearQuarter1'].str.replace("\n","")
total_rapid_forecast['YearQuarter1'] = total_rapid_forecast['YearQuarter1'].str.replace("(","")
total_rapid_forecast['YearQuarter1'] = total_rapid_forecast['YearQuarter1'].str.replace(")","")
total_rapid_forecast['YearQuarter1'] = total_rapid_forecast['YearQuarter1'].str.replace("Total Charging Devices","")
total_rapid_forecast['YearQuarter1'] = total_rapid_forecast['YearQuarter1'].str.replace("Jan-22 2", "Jan-22")


total_rapid_forecast = total_rapid_forecast.rename(columns = {'YearQuarter1':'YearQuarter','Local Authority / Region Code':'RegionCode',
                                       'Local Authority / Region Name_x':'RegionName'})
total_rapid_forecast['YearQuarter'].unique()
total_rapid_forecast['YearQuarter'] = total_rapid_forecast['YearQuarter'].str.strip()
total_rapid_forecast['YearQuarter'] = total_rapid_forecast['YearQuarter'].str.replace("July","Jul")
total_rapid_forecast['YearQuarter'] = pd.to_datetime(total_rapid_forecast['YearQuarter'], format='%b-%y')
total_rapid_forecast['YearQuarter']

total_rapid_forecast.head()
total_rapid_forecast.to_csv('total_rapid_forecast.csv')
total_rapid_forecast[total_rapid_forecast['RegionName']=='Barnet']
ev_result
rapid_charging_result
# correlation
corr_matrix = total_rapid_forecast.corr()

# correlation - heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.show()




# Filter data to include only boroughs with charging stations
data = total_rapid_forecast[total_rapid_forecast['TotalChargingDevices'] > 0]

# Filter data to include only boroughs with EVs
data = data[data['TotalEV'] > 0]
data[data['RegionName']=='Haringey']
data.head()


# Group data by borough
data_grouped = data.groupby('RegionName').sum().reset_index()
data_grouped
# Calculate the percentage reduction in charging stations
reduction = 0.1
data_grouped['charging_stations_reduced'] = data_grouped['TotalChargingDevices'] * (1 - reduction)

# Create a new dataframe with the date range and boroughs
dates = pd.date_range(start='2023-05-01', end='2024-09-30', freq='M')
dates
boroughs = data_grouped['RegionName'].unique()
boroughs
forecast_data = pd.DataFrame({'ds': [], 'borough': [], 'yhat': []})

# Loop through each borough and make a forecast
for borough in boroughs:
    # Filter data for the current borough
    data_filtered = data[data['RegionName'] == borough].copy()
    data_filtered['YearQuarter'] = pd.to_datetime(data_filtered['YearQuarter'])
    data_filtered['year'] = data_filtered['YearQuarter'].dt.year
    data_filtered['quarter'] = data_filtered['YearQuarter'].dt.quarter
    
    # Calculate the percentage reduction in charging stations for the current borough
    charging_stations_reduced = data_grouped[data_grouped['RegionName'] == borough]['charging_stations_reduced'].values[0]
    
    # Create a new dataframe for the current borough
    borough_data = pd.DataFrame({'ds': dates, 'y': [0] * len(dates)})
    borough_data['RegionName'] = borough
    # borough_data
    # Fill in the EV data
    for index, row in data_filtered.iterrows():
        year = row['year']
        quarter = row['quarter']
        date_index = (year - 2019) * 4 + quarter - 1
        borough_data.at[date_index, 'y'] = row['TotalEV']
    
    # Fit the Prophet model
    model = Prophet()
    model.fit(borough_data)
    borough_data
    # Make a forecast for the current borough
    future = model.make_future_dataframe(periods=12, freq='M')
    future['RegionName'] = borough
    future['charging_stations_reduced'] = charging_stations_reduced
    forecast = model.predict(future)
    forecast['borough'] = borough

    # Add the forecast to the forecast data
    forecast_data = pd.concat([forecast_data, forecast[['ds', 'borough', 'yhat']]], axis=0)


# Print the forecast data
print(forecast_data)

# Filter data to include only boroughs with EVs and charging stations
data = total_rapid_forecast[(total_rapid_forecast['TotalChargingDevices'] > 0) & (total_rapid_forecast['TotalEV'] > 0)]
data
# Convert YearQuarter column to datetime and add year and quarter columns
data['YearQuarter'] = pd.to_datetime(data['YearQuarter'])
data['year'] = data['YearQuarter'].dt.year
data['quarter'] = data['YearQuarter'].dt.quarter

# Calculate the percentage reduction in charging stations (the ratio of rapid charging station is always 30% out of the total station)

reduction = 0.3
data_grouped = data.groupby('RegionName').sum().reset_index()
data_grouped['charging_stations_reduced'] = data_grouped['TotalChargingDevices'] * (1 - reduction)

# Create a function to fit a Prophet model to the data for a borough and make a forecast
def fit_prophet(data):
    # Create a new dataframe for the current borough
    borough_data = pd.DataFrame({'ds': dates, 'y': [0] * len(dates)})
    borough_data['RegionName'] = data['RegionName'].iloc[0]

    # Fill in the EV data
    for index, row in data.iterrows():
        year = row['year']
        quarter = row['quarter']
        date_index = (year - 2019) * 4 + quarter - 1
        borough_data.at[date_index, 'y'] = row['TotalEV']

    # Fit the Prophet model
    model = Prophet()
    model.fit(borough_data)

    # Make a forecast for the current borough
    future = model.make_future_dataframe(periods=12, freq='M')
    future['RegionName'] = borough_data['RegionName'].iloc[0]
    future['charging_stations_reduced'] = data_grouped[data_grouped['RegionName'] == borough_data['RegionName'].iloc[0]]['charging_stations_reduced'].iloc[0]
    forecast = model.predict(future)

    return forecast[['ds', 'RegionName', 'yhat']]

# Create a new dataframe with the date range and boroughs
dates = pd.date_range(start='2023-05-01', end='2024-09-30', freq='M')
boroughs = data_grouped['RegionName'].unique()

# Create a new dataframe with the date range and boroughs
forecast_data = pd.DataFrame()
forecast_data
for borough in boroughs:
    # Filter data for the current borough
    data_filtered = data[data['RegionName'] == borough].copy()
    data_filtered['YearQuarter'] = pd.to_datetime(data_filtered['YearQuarter'])
    data_filtered['year'] = data_filtered['YearQuarter'].dt.year
    data_filtered['quarter'] = data_filtered['YearQuarter'].dt.quarter
    
    # Calculate the percentage reduction in charging stations for the current borough
    charging_stations_reduced = data_grouped.loc[data_grouped['RegionName'] == borough, 'charging_stations_reduced'].values[0]
    
    # Create a new dataframe for the current borough
    borough_data = pd.DataFrame({'ds': dates, 'y': [0] * len(dates)})
    borough_data['RegionName'] = borough
    borough_data
    # Fill in the EV data
    for index, row in data_filtered.iterrows():
        year = row['year']
        quarter = row['quarter']
        date_index = (year - 2019) * 4 + quarter - 1
        borough_data.at[date_index, 'y'] = row['TotalEV']
    
    # Fit the Prophet model
    model = Prophet()
    model.fit(borough_data)
    borough_data
    # Make a forecast for the current borough
    future = model.make_future_dataframe(periods=12, freq='M')
    future['RegionName'] = borough
    future['charging_stations_reduced'] = charging_stations_reduced
    forecast = model.predict(future)
    forecast['borough'] = borough
    
    
    # Add the forecast to the forecast data
    forecast_data = pd.concat([forecast_data, forecast[['ds', 'borough', 'yhat']]], axis=0)


forecast_data.keys()
forecast_data = forecast_data[['ds', 'borough', 'yhat']]
forecast_data['borough'].unique()
forecast_data[forecast_data['borough']== 'Kensington and Chelsea']

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

for borough in boroughs:
    borough_forecast = forecast_data[forecast_data['borough'] == borough]
    ax.plot(borough_forecast['ds'], borough_forecast['yhat'], label=borough)

ax.set_title("Forecast EVs - Rapid charging staions are 30%")
ax.set_xlabel("Date")
ax.set_ylabel("Forecasted Value")
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.show()




# choose a region to plot
region = 'Barnet'

# filter the data to the selected region
region_data = forecast_data[forecast_data['borough'] == region]
print(region_data.head())
region_data['ds']
total_rapid_forecast[total_rapid_forecast['RegionName']=='Barnet']
region_data[region_data['borough']=='Barnet']

# Define a function to fit the model and make forecasts for each group
def fit_and_forecast(group):
    # Rename columns to 'ds' and 'y'
    group = group.rename(columns={'YearQuarter': 'ds', 'TotalEV': 'y'})
    # Fit the model to the group's data
    model = Prophet()
    model.fit(group)
    # Make a dataframe with the future dates
    future = model.make_future_dataframe(periods=12, freq='M')
    # Make the forecasts for the group's data
    forecast = model.predict(future)
    # Add a column with the region code to the forecasts
    forecast['RegionName'] = group['RegionName'].unique()[0]
    # Return the forecasts for the group
    return forecast

# Apply the function to each group in grouped_data
forecasts_by_region = grouped_data.apply(fit_and_forecast).reset_index(drop=True)

# Plot the forecasts for a selected region
region = 'Newham'
region_data = forecasts_by_region[forecasts_by_region['RegionName'] == region]
plt.plot(region_data['ds'], region_data['yhat'])
plt.title(f"Forecast for {region}")
plt.xlabel("Date")
plt.ylabel("Forecasted Value")
plt.show()


import matplotlib.pyplot as plt

# Get unique borough names
boroughs = forecasts_by_region['RegionName'].unique()
boroughs
# Create a plot with subplots for each borough
fig, axs = plt.subplots(len(boroughs), figsize=(10, 40))

# Loop through each borough and plot the forecasted values
for i, borough in enumerate(boroughs):
    # Filter the data for the current borough
    region_data = forecasts_by_region[forecasts_by_region['RegionName'] == borough]
    # Plot the data on the corresponding subplot
    axs[i].plot(region_data['ds'], region_data['yhat'])
    axs[i].set_title(f"Forecast for {borough}")
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("Forecasted Value")

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()




import pandas as pd
from fbprophet import Prophet

# when the number of total charging station increased by 20% 

df = pd.read_csv('total_rapid_forecast.csv')
df.keys()
df = df.drop(columns = 'Unnamed: 0')
df.head()
# 데이터프레임에서 필요한 컬럼만 추출
data_for_total= df[['YearQuarter', 'RegionName', 'TotalChargingDevices', 'TotalEV']]
data_for_total.head()
# 총 전기차 충전소의 수를 30% 증가시킨다고 가정
data_for_total['TotalChargingDevices'] = data_for_total['TotalChargingDevices'] * 1.3

# 데이터 전처리: YearQuarter을 datetime 타입으로 변환하고 year, quarter 컬럼 추가
data_for_total['YearQuarter'] = pd.to_datetime(data_for_total['YearQuarter'])
data_for_total['year'] = data_for_total['YearQuarter'].dt.year
data_for_total['quarter'] = data_for_total['YearQuarter'].dt.quarter
data_for_total.head()
# 각 지역별로 모델을 학습하고 예측 결과를 리스트에 저장
forecast_list = []
for region in data_for_total['RegionName'].unique():
    # 해당 지역의 데이터 추출
    region_data = data_for_total[data_for_total['RegionName'] == region][['YearQuarter', 'TotalEV']]
    region_data = region_data.rename(columns={'YearQuarter': 'ds', 'TotalEV': 'y'})
    
    # Prophet 모델 생성 및 학습
    model_total = Prophet()
    model_total.fit(region_data)
    
    # 2년치 예측
    future_2 = model_total.make_future_dataframe(periods=24, freq='Q')
    forecast_2 = model_total.predict(future)
    
    # 결과 데이터프레임에 지역명 추가
    forecast_2['RegionName'] = region
    
    # 결과 데이터프레임을 리스트에 추가
    forecast_list.append(forecast_2[['RegionName', 'ds', 'yhat']])
    
# 각 지역별 예측 결과를 하나의 데이터프레임으로 합침
forecast_df = pd.concat(forecast_list)

# 결과 출력
print(forecast_df)

# choose a region to plot
region = 'Haringey'

# filter the data to the selected region
region_df = forecast_df[forecast_df['RegionName'] == region]
print(region_df.head())
region_df['ds']
total_rapid_forecast[total_rapid_forecast['RegionName']=='Haringey']
region_df[region_df['RegionName']=='Haringey']


# Calculate the percentage reduction in charging stations increase by  30% 
data_grouped = data.groupby('RegionName').sum().reset_index()
data_grouped['charging_stations_increase'] = data_grouped['TotalChargingDevices'] + (data_grouped['TotalChargingDevices']* 0.3)

# Get unique borough names
boroughs = forecast_df['RegionName'].unique()
boroughs

for borough in boroughs:
    # Filter data for the current borough
    data_filtered = data[data['RegionName'] == borough].copy()
    data_filtered['YearQuarter'] = pd.to_datetime(data_filtered['YearQuarter'])
    data_filtered['year'] = data_filtered['YearQuarter'].dt.year
    data_filtered['quarter'] = data_filtered['YearQuarter'].dt.quarter
    
    # Calculate the percentage reduction in charging stations for the current borough
    charging_stations_increase = data_grouped.loc[data_grouped['RegionName'] == borough, 'charging_stations_increase'].values[0]
    
    # Create a new dataframe for the current borough
    borough_data = pd.DataFrame({'ds': dates, 'y': [0] * len(dates)})
    borough_data['RegionName'] = borough
    borough_data
    # Fill in the EV data
    for index, row in data_filtered.iterrows():
        year = row['year']
        quarter = row['quarter']
        date_index = (year - 2019) * 4 + quarter - 1
        borough_data.at[date_index, 'y'] = row['TotalEV']
    
    # Fit the Prophet model
    model = Prophet()
    model.fit(borough_data)
    borough_data
    # Make a forecast for the current borough
    future = model.make_future_dataframe(periods=12, freq='M')
    future['RegionName'] = borough
    future['charging_stations_increase'] = charging_stations_increase
    forecast = model.predict(future)
    forecast['borough'] = borough
    
    
    # Add the forecast to the forecast data
    forecast_data = pd.concat([forecast_data, forecast[['ds', 'borough', 'yhat']]], axis=0)


forecast_data.keys()
forecast_data = forecast_data[['ds', 'borough', 'yhat']]
forecast_data['borough'].unique()
forecast_data

fig, ax = plt.subplots(figsize=(10, 6))

for borough in boroughs:
    borough_forecast = forecast_data[forecast_data['borough'] == borough]
    ax.plot(borough_forecast['ds'], borough_forecast['yhat'], label=borough)

ax.set_title("Forecast EVs - Total charging station increases by 30%")
ax.set_xlabel("Date")
ax.set_ylabel("Forecasted Value")
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.show()