import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import string
import re
import statsmodels.api as sm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve , auc
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.svm import SVC
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.pipeline import make_pipeline






ec_df = pd.read_csv('total_final_energy_consumption.csv')

print(e_df.head())


is_Newham = e_df['Area'] == 'Newham'
f_c = e_df['Fuel'] == 'Coal'
e_s = e_df['Sector'] == 'Domestic'
e_m = e_df['Measurement'] == 'GWh'

b_c_2005 = e_df[is_Newham & f_c &  e_s] 
print(b_c_2005)
pd.set_option('display.max_rows', None)
with open('Newham_domestic.csv', 'w', encoding= 'utf=8') as Newham_coal_domestic:
        print(b_c_2005, file = Newham_coal_domestic)

n_d_c_df = pd.read_csv('Newham_domestic_GWh.csv')
sns.lineplot( x = 'Year', y = 'Value', data = n_d_c_df)
plt.show()

e_df.loc[e_df['Value']== '..', 'Value'] = 0
e_df.loc[e_df['Value']== '-', 'Value'] = 0

df = e_df.groupby(['Area','Year','Fuel','Sector','Measurement'])['Value'].mean()
df.groupby('Area')


df =  e_df.groupby(['Area','Year','Fuel','Sector','Measurement'])['Value'].mean().unstack()
print(df.head(30))

with open('area.csv', 'w', encoding= 'utf=8') as data:
        print(df, file = data)

sns.lineplot(x= 'Year', y = 'Area', data = df, hue= 'Fuel')
plt.show()


data = pd.read_csv('./domestic-E09000025-Newham/certificates.csv')
data.shape
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 123801)
print(data.head())
data.info()
data.isnull().sum()
data.isnull().sum()/len(data)*100 # 결측치를 퍼센트로 변환 
data.isnull().mean().sort_values(ascending= False)



# time 
data['INSPECTION_DATE']
data['INSPECTION_YEAR'] = pd.DatetimeIndex(data['INSPECTION_DATE']).year
print(data['INSPECTION_YEAR'])
sorted(data['INSPECTION_YEAR'].unique())
new_df = data[['CONSTRUCTION_AGE_BAND', 'PROPERTY_TYPE', 'LMK_KEY']]
new_df.isnull().sum()
new_df.dropna(inplace=True)
new_df.isnull().sum()
new_df.head()
new_df['CONSTRUCTION_AGE_BAND'].value_counts()
nodata = new_df['CONSTRUCTION_AGE_BAND'].isin(['NO DATA!'])
new_df[~nodata].head()
new_df= new_df[~nodata]
new_df['CONSTRUCTION_AGE_BAND'].value_counts()
invalid = new_df['CONSTRUCTION_AGE_BAND'].isin(['INVALID!'])
new_df = new_df[~invalid]
new_df['CONSTRUCTION_AGE_BAND'].value_counts()
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.strip()
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='England and Wales:', repl='', regex=False)
new_df['CONSTRUCTION_AGE_BAND'].value_counts()
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.strip()
new_df['CONSTRUCTION_AGE_BAND'].value_counts()
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='onwards', repl='', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='before', repl='-', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2007-2011', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2021', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2022', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2018', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2019', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2012', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2020', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2013', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2017', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2016', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2015', repl='2007', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='2007', repl='2007-2022', regex=False)
new_df['CONSTRUCTION_AGE_BAND'] = new_df['CONSTRUCTION_AGE_BAND'].str.replace(pat='- 1900', repl='0000-1900', regex=False)
new_df.info

h_df = new_df[new_df['PROPERTY_TYPE'].str.contains('House')]
h_df['CONSTRUCTION_AGE_BAND'].value_counts()
h_df['PROPERTY_TYPE'].value_counts()
h_df['CONSTRUCTION_AGE_BAND'].describe()
f_df = new_df[new_df['PROPERTY_TYPE'].str.contains('Flat')]
f_df['PROPERTY_TYPE'].value_counts()
f_df['CONSTRUCTION_AGE_BAND'].value_counts()
f_df['CONSTRUCTION_AGE_BAND'].describe()
m_df = new_df[new_df['PROPERTY_TYPE'].str.contains('Maisonette')]
m_df['CONSTRUCTION_AGE_BAND'].value_counts()
m_df['PROPERTY_TYPE'].value_counts()
m_df.describe()
b_df = new_df[new_df['PROPERTY_TYPE'].str.contains('Bungalow')]
b_df['CONSTRUCTION_AGE_BAND'].value_counts()
b_df['PROPERTY_TYPE'].value_counts()
b_df.describe()

# building references and energy rateings and the built form
data['BUILDING_REFERENCE_NUMBER']
data['BUILDING_REFERENCE_NUMBER'].value_counts(ascending= False)
data['CURRENT_ENERGY_RATING'].value_counts(ascending= False)
data['POTENTIAL_ENERGY_RATING'].value_counts(ascending= False)
data[data['BUILT_FORM'].isnull()]
data.dropna(subset=('BUILT_FORM'), inplace= True)
data['BUILT_FORM'].value_counts()
data.shape
mask = data['BUILT_FORM'].isin(['NO DATA!'])
data[~mask].head()
df = data[~mask]
df.shape
df.head()
df['BUILT_FORM'].value_counts()
df['CURRENT_ENERGY_RATING'].value_counts()
df['POTENTIAL_ENERGY_RATING'].value_counts()
built = df[['BUILT_FORM', 'LMK_KEY' ]]
built.head()

h_df= pd.merge(h_df,built,  how = 'inner' ,on = 'LMK_KEY')
h_df.describe()
h_df['CONSTRUCTION_AGE_BAND'].value_counts()
f_df= pd.merge(f_df,built,  how = 'inner' ,on = 'LMK_KEY')
f_df.describe()
f_df['CONSTRUCTION_AGE_BAND'].value_counts()
m_df= pd.merge(m_df,built,  how = 'inner' ,on = 'LMK_KEY')
m_df.describe()
m_df['CONSTRUCTION_AGE_BAND'].value_counts()
b_df= pd.merge(b_df,built,  how = 'inner' ,on = 'LMK_KEY')
b_df.describe()
b_df['CONSTRUCTION_AGE_BAND'].value_counts()
h_df.shape



# 중복값
len(df)
df.index
df['LMK_KEY'].nunique()
df.describe()
df['BUILT_FORM'].value_counts()
df['BUILT_FORM'].index

# current eneregy rate
df.groupby('BUILT_FORM')['CURRENT_ENERGY_EFFICIENCY'].mean()
df.groupby('BUILT_FORM')['CURRENT_ENERGY_EFFICIENCY'].transform('mean')
df.shape
df['ENERGY_EFFICIENCY_BY_BUILT_FORM_MEAN'] = df.groupby('BUILT_FORM')['CURRENT_ENERGY_EFFICIENCY'].transform('mean')
df.head()
# df['ENERGY_EFFICIENCY_BY_BUILT_FORM_COUNT'] = df.groupby('BUILT_FORM')['CURRENT_ENERGY_EFFICIENCY'].transform('count')
# df['ENERGY_EFFICIENCY_BY_BUILT_FORM_COUNT'].head(10)
e_df = df[['ENERGY_EFFICIENCY_BY_BUILT_FORM_MEAN','CURRENT_ENERGY_EFFICIENCY','BUILT_FORM','CURRENT_ENERGY_RATING']]
e_df.head()
e_df.drop(columns=['COUNT'], inplace= True)
e_df.keys()

df.groupby(['BUILT_FORM','CURRENT_ENERGY_RATING'])['CURRENT_ENERGY_RATING'].count()
df['COUNT'] = df.groupby(['BUILT_FORM','CURRENT_ENERGY_RATING'])['CURRENT_ENERGY_RATING'].transform('count')
df.head()
e_df = df[['ENERGY_EFFICIENCY_BY_BUILT_FORM_MEAN','CURRENT_ENERGY_EFFICIENCY','BUILT_FORM','CURRENT_ENERGY_RATING',\
        'COUNT','LMK_KEY']]
e_df = e_df.sort_values(by =['CURRENT_ENERGY_RATING'],ascending=[True])
e_df.describe()

#######
print(len(e_df), e_df['BUILT_FORM'].nunique())
built_df = pd.DataFrame(e_df['BUILT_FORM'].value_counts())
built_df.reset_index(inplace=True)
# the percentage of the building types in New Ham
fig = px.pie(built_df, values= 'BUILT_FORM', names= 'index',title='Each Built form percentage and numbers in Newham',\
        hole=.3,color="BUILT_FORM")
fig.update_traces(textposition='inside', textinfo='percent+value')
fig.show()
# current energy ratings by building types 
fig1 = px.histogram(e_df, x="CURRENT_ENERGY_RATING", color="BUILT_FORM",\
        labels={'CURRENT_ENERGY_RATING':'Current energy rating (A-G) in Newham',\
               'BUILT_FORM': 'Built form' }, text_auto= True,\
               title= 'Each energy rating mdae up by (?) each building built form')
fig1.update_layout(yaxis_title='Count')
fig1.show()

# current energy ratings by building types 
sns.histplot( x = 'CURRENT_ENERGY_RATING', hue= 'BUILT_FORM', multiple="stack", data = e_df)
plt.show()

# potential energy ratings by building types
df.groupby('BUILT_FORM')['POTENTIAL_ENERGY_EFFICIENCY'].mean()
len(df.groupby('BUILT_FORM')['POTENTIAL_ENERGY_EFFICIENCY'].transform('mean'))
df['POTENTIAL_ENERGY_EFFICIENCY_MEAN'] = df.groupby('BUILT_FORM')['POTENTIAL_ENERGY_EFFICIENCY'].transform('mean')
df.shape
ep_df = df[['POTENTIAL_ENERGY_EFFICIENCY_MEAN','POTENTIAL_ENERGY_EFFICIENCY','BUILT_FORM','POTENTIAL_ENERGY_RATING']]
ep_df.head()
ep_df = ep_df.sort_values(by =['POTENTIAL_ENERGY_RATING'],ascending=[True])

# potential energy ratings by building types 
sns.histplot( x = 'POTENTIAL_ENERGY_RATING', hue= 'BUILT_FORM', multiple="stack", data = ep_df)
plt.show()
fig2 = px.histogram(ep_df, x="POTENTIAL_ENERGY_RATING", color="BUILT_FORM",\
        labels={'POTENTIAL_ENERGY_RATING':'Expected energy rating (A-G) in Newham',\
               'BUILT_FORM': 'Built form' }, text_auto= True,\
               title= 'Expected each energy rating mdae up by (?) each building built form')
fig2.update_layout(yaxis_title='Count')
fig2.show()


# clean windows data
df.shape
df['LMK_KEY'].nunique()
win_df = df[['LMK_KEY','WINDOWS_DESCRIPTION','WINDOWS_ENERGY_EFF']]
win_df.info
win_df.isnull().sum()
win_df.dropna(inplace= True)
win_df.isnull().sum()
sorted(win_df['WINDOWS_DESCRIPTION'].unique())
win_df['WINDOWS_DESCRIPTION'].value_counts()
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.strip()
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.replace(pat='ing', repl='ed', regex = False)
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.replace(pat='Sedle', repl='Single', regex = False)
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.replace(pat='glazeddouble', repl='glazed double', regex = False)
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.replace(pat='Fully', repl='Full', regex = False)
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.replace(pat='Gwydrau dwbl llawn', repl='Full double glazed', regex = False)
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.capitalize()
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.replace(pat='  ', repl='', regex = False)
win_df['WINDOWS_DESCRIPTION'] = win_df['WINDOWS_DESCRIPTION'].str.replace(pat='Mostlydouble', repl='Mostly double', regex = False)
type(win_df)
win_df.head()

#clean PROPERTY_TYPE 

p_df = df[['LMK_KEY','PROPERTY_TYPE']]
p_df.info
p_df.isnull().sum()
p_df['PROPERTY_TYPE'].value_counts()

#clean WALLS_DESCRIPTION

w_df = df[['LMK_KEY','WALLS_DESCRIPTION','WALLS_ENERGY_EFF']]
w_df.info
w_df.isnull().sum()
w_df.dropna(inplace=True)
w_df.isnull().sum()
w_df.shape
w_df['WALLS_DESCRIPTION'].value_counts()
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='  ', repl='', regex = False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='e0', repl='e 0', regex = False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='W/m-¦K', repl='W/m²K', regex=False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='W/m&#0178;K', repl='W/m²K', regex = False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='W/mÂ²K', repl='W/m²K', regex = False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='W/m??K', repl='W/m²K', regex = False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='W/m?K', repl='W/m²K', regex = False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='=', repl='', regex = False)
w_df['WALLS_DESCRIPTION'] = w_df['WALLS_DESCRIPTION'].str.replace(pat='Briciau solet, fel y?u hadeiladwyd, dim inswleiddio (rhagdybiaeth)'\
        , repl='Solid bricks, as built, no insulation (assumption)', regex = False)
w_df['WALLS_DESCRIPTION'].value_counts()
w_df.head(10)


#clean ROOF_DESCRIPTION
r_df = df[['LMK_KEY','ROOF_DESCRIPTION','ROOF_ENERGY_EFF']]
r_df.isnull().sum()
r_df.dropna(inplace=True)
r_df.isnull().sum()
r_df['ROOF_DESCRIPTION'].value_counts()
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.strip()  # 앞 뒤 공백을 제거
r_df['ROOF_DESCRIPTION'].value_counts()
units = ['W/m-¦K', 'W/m&#0178;K', 'W/mÂ²K','W/m??K']
for unit in units:
        r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace(unit,'W/m²K')
r_df['ROOF_DESCRIPTION'].value_counts()
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].\
        str.replace(pat='Ar oleddf, 300+ mm mm o inswleiddio yn y llofft', \
        repl='pitched, 300+ mm loft insulation ', regex = False)
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace(',','')
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace('?','²')
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace('²²','²')
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace('=','')
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.capitalize()

r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.strip()  # 앞 뒤 공백을 제거
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace('  ',' ')
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace('mm',' mm')
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].str.replace('  ',' ')
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].\
        str.replace(pat='Average thermal transmittance 0.1 w/m²k', \
        repl='Average thermal transmittance 0.10 w/m²k', regex = False)
r_df['ROOF_DESCRIPTION'] = r_df['ROOF_DESCRIPTION'].\
        str.replace(pat='Average thermal transmittance 0.2 w/m²k', \
        repl='Average thermal transmittance 0.20 w/m²k', regex = False)
r_df['ROOF_DESCRIPTION'].value_counts()
r_df['ROOF_DESCRIPTION'].nunique()

in_df = pd.merge(e_df, w_df, how = 'inner' ,on = 'LMK_KEY')
in_df.head()

in_df1 = pd.merge(in_df, win_df, how = 'inner' ,on = 'LMK_KEY')

in_df2 = pd.merge(in_df1, r_df, how = 'inner' ,on = 'LMK_KEY')
in_df3 = pd.merge(in_df2, p_df, how = 'inner' ,on = 'LMK_KEY')
in_df3.head(30)
data['BUILT_FORM'].unique()
### energy efficiency by property
sns.catplot(x = 'PROPERTY_TYPE', y ='CURRENT_ENERGY_EFFICIENCY' , \
        data= in_df3, kind= 'box', hue = 'BUILT_FORM' , notch=True,\
        showfliers=False,\
                 palette= {'Mid-Terrace':'#636efa', \
                 'End-Terrace':'#ef553c','Detached':'#07cb96',\
                  'Semi-Detached':'#ab63f9', 'Enclosed End-Terrace': '#25d1ef','Enclosed Mid-Terrace': '#fda15b'})
plt.title('Title')
plt.xlabel('Property types')
plt.ylabel('Current energy efficiency')
plt.show()
in_df3['BUILT_FORM'].unique()
# fig3 = go.Figure()
# fig3.add_trace(go.Box(
#     x= in_df3['PROPERTY_TYPE'],
#     y=in_df3['CURRENT_ENERGY_EFFICIENCY'],
#     name='Mid-Terrace',
#     notched=True,
#     marker_color='#636efa'
# ))
# fig3.add_trace(go.Box(
#         x= in_df3['PROPERTY_TYPE'],
#     y=in_df3['CURRENT_ENERGY_EFFICIENCY'],
#     name ='Semi-Detached',
#     marker_color='#ab63f9'
# ))
# fig3.add_trace(go.Box(
#     x= in_df3['PROPERTY_TYPE'],
#     y=in_df3['CURRENT_ENERGY_EFFICIENCY'],
#     name='End-Terrace',
#     marker_color='#df553c'
# ))
# fig3.add_trace(go.Box(
#     x= in_df3['PROPERTY_TYPE'],
#     y=in_df3['CURRENT_ENERGY_EFFICIENCY'],
#     name='Detached',
#     marker_color='#07cb86'
# ))
# fig3.add_trace(go.Box(
#     x= in_df3['PROPERTY_TYPE'],
#     y=in_df3['CURRENT_ENERGY_EFFICIENCY'],
#     name='Enclosed Mid-Terrace',
#     marker_color='#fda15b'
# ))
# fig3.add_trace(go.Box(
#     x= in_df3['PROPERTY_TYPE'],
#     y=in_df3['CURRENT_ENERGY_EFFICIENCY'],
#     name='Enclosed End-Terrace',
#     marker_color='#25d1ef'
# ))

# fig3.update_layout(
#     xaxis=dict(title='normalized moisture', zeroline=False),
#     boxmode='group', xaxis_tickangle=0
# )
# fig3.show()

### index
in_df3.keys()

### one-hot-encoding using get_dummies 
def dummy_data(data, columns):
        for column in columns:
                data = pd.concat([data, pd.get_dummies(data[column], prefix = column)], axis= 1)
                data = data.drop(column, axis=1)
        return data

dummy_columns = ['BUILT_FORM', 'WALLS_DESCRIPTION', \
       'WINDOWS_DESCRIPTION', 'ROOF_DESCRIPTION',\
        'PROPERTY_TYPE']
dummy_df = dummy_data(in_df3,dummy_columns)

dummy_df.keys()
### data and target  ----- > Decision tree
X = dummy_df.drop(['CURRENT_ENERGY_RATING','COUNT','ENERGY_EFFICIENCY_BY_BUILT_FORM_MEAN',\
        'CURRENT_ENERGY_EFFICIENCY','LMK_KEY','WALLS_ENERGY_EFF',\
                'WINDOWS_ENERGY_EFF', 'ROOF_ENERGY_EFF'], axis=1)
X.head()

y = in_df3['CURRENT_ENERGY_RATING']
y.head()

### class(target) label encoding
class_le = LabelEncoder()
y = class_le.fit_transform(y) # Alphabetical order
y

### Train and Test
X_train, X_test, y_train, y_test = \
        train_test_split(X,y,
        test_size =0.3,
        random_state=100000,
        stratify=y)

tree = DecisionTreeClassifier(criterion= 'gini',\
        max_depth= None,
        random_state=100000)
tree.fit(X_train, y_train)

# predicted values
y_pred = tree.predict(X_test)
y_pred

# probability for forecasting
y_pred_p = tree.predict_proba(X_test)
y_pred_p

# 정오분류표로 검정 confusion matrix
confmat = pd.DataFrame(confusion_matrix(y_test, y_pred),\
        index = ['True[0]', 'True[1]','True[2]','True[3]','True[4]','True[5]','True[6]'],
        columns=['Predict[0]', 'Predict[1]','Predict[2]','Predict[3]','Predict[4]','Predict[5]','Predict[6]'])
confmat

sys.stdout = open('Decision.txt','w')
print('Classification Report')
print(classification_report(y_test, y_pred))
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")
print('Classification Report')
print(classification_report(y_test, y_pred))

## accuracy and precission and whatever
print('The number of samples with a wrong prediction : %d' %(y_test != y_pred).sum())
print('Accuracy Score: %.3f' %accuracy_score(y_test,y_pred))
# print('Precision Score %.3f' %precision_score) # When I have 2 classes
# print('Precision Score %.3f' %precision_score(y_test, y_pred, average=None))
####will return the precision scores for each class
print('Precision Score: %.3f' %precision_score(y_test, y_pred, average='micro'))
#### return the total ratio of tp/(tp + fp)
#### The pos_label argument will be ignored if you choose another average option than binary
# print('Recall Score %.3f' %recall_score (y_true=y_test, y_pred=y_pred)) # When I have 2 classes
print('Recall Score: %.3f' %recall_score (y_true=y_test, y_pred=y_pred, average='micro'))
print('F1: %.3f'  %f1_score(y_true=y_test, y_pred= y_pred,average='micro'))
# print('F1: %.3f'  %f1_score(y_true=y_test, y_pred= y_pred)) # When I have 2 classes

# # Decision tree
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from PIL import Image
import cairo
import cairosvg
feature_names = X.columns.tolist()
target_name = np.array(['A', 'B','C','D','E','F','G'])
dot_data = export_graphviz(tree,\
        filled =True, rounded = True, class_names= target_name,
        feature_names= feature_names,
        out_file = None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf('De_tree.pdf')
dt_graph = graph_from_dot_data(dot_data)
Image(dt_graph.create_png())

# ### majority voting
# logistic = LogisticRegression()
# tree = DecisionTreeClassifier()
# knn = KNeighborsClassifier()

# voting_estimators = [('logistic', logistic), ('tree', tree), ('knn', knn)]

# voting = VotingClassifier(estimators = voting_estimators,
#                                 voting = 'soft')
# clf_labels = ['Logistic regression', 'Decision Tree', 'KNN', 'Majority voting']

# all_clf = [logistic, tree, knn, voting]

# ##AUC

# clf_labels = ['Logistic regression', 'Decision tree', 'KMN', 'Majority voting']
# all_clf = [logistic, tree, knn, voting]
# for clf, label in zip(all_clf, clf_labels) :
#         scores = cross_val_score(estimator=clf,X=X_train,y=y_train, cv=10,scoring='roc_auc')
#         print ("ROC AUC: 80.31 (+/- 90.31) [%]" % (scores. mean(), scores.std(), label))

## 데이터 간 상관관계
in_df3.head()
sns.pairplot(data[['HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT', 'HOT_WATER_COST_CURRENT']])
plt.show()

data['intercept'] = 1
lm_heating = sm.OLS(data['CURRENT_ENERGY_EFFICIENCY'], data[['intercept','MAINHEAT_ENERGY_EFF']])
lm_heating = sm.OLS(data['CURRENT_ENERGY_EFFICIENCY'], data[['intercept','HEATING_COST_CURRENT']])
lm_hotwater = sm.OLS(data['CURRENT_ENERGY_EFFICIENCY'], data[['intercept','HOT_WATER_COST_CURRENT']])
lm = sm.OLS(data['CURRENT_ENERGY_EFFICIENCY'], data[['intercept', 'HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT', 'HOT_WATER_COST_CURRENT']])
results1 = lm_heating.fit()
results2 = lm_heating.fit()
results3 = lm_hotwater.fit()
results = lm.fit()
sys.stdout = open('VIF.txt','w')
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")
results1.summary()
results2.summary()
results3.summary()
results.summary()

### VIF
y, X = patsy.dmatrices('CURRENT_ENERGY_EFFICIENCY ~ LIGHTING_COST_CURRENT + HEATING_COST_CURRENT + HOT_WATER_COST_CURRENT', data, return_type = 'dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
vif

### dummy variables
# Regression Results  by property type = Compare with Bungalow, Flat's energy efficiency average is 10ish higher
in_df3.head()
p_dummy = in_df3['PROPERTY_TYPE']
p_dummy.unique()
dumy_df = pd.get_dummies(in_df3.PROPERTY_TYPE)
dumy_df.head()
dumy_df.keys()
dumy_df['intercept'] = 1
property = sm.OLS(in_df3['CURRENT_ENERGY_EFFICIENCY'], dumy_df[['intercept', 'Flat', 'House', 'Maisonette']]).fit().summary()
sys.stdout = open('property.txt','w')
print(property)
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

# Regression Results by BUILT_FORM
b_dummy = in_df3['BUILT_FORM']
b_dummy.unique()
dumy_df_b = pd.get_dummies(in_df3.BUILT_FORM)
dumy_df_b.head()
dumy_df_b.keys()
dumy_df_b['intercept'] = 1
built = sm.OLS(in_df3['CURRENT_ENERGY_EFFICIENCY'], \
        dumy_df_b[['intercept', 'Enclosed End-Terrace', 'Enclosed Mid-Terrace',
       'End-Terrace', 'Mid-Terrace', 'Semi-Detached']]).fit().summary()
sys.stdout = open('built.txt','w')
print(built)
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

## efficiency change to number 
w_eff = data[['LMK_KEY','WINDOWS_ENERGY_EFF']]
w_eff.head()
w_eff.isnull().sum()
w_eff.dropna(inplace= True)
w_eff.isnull().sum()
w_eff.value_counts()
w_eff['WINDOWS_ENERGY_EFF'] = w_eff['WINDOWS_ENERGY_EFF'].str.replace('Very Good', 'A',regex=False)
w_eff['WINDOWS_ENERGY_EFF'] = w_eff['WINDOWS_ENERGY_EFF'].str.replace('Good', 'B',regex=False)
w_eff['WINDOWS_ENERGY_EFF'] = w_eff['WINDOWS_ENERGY_EFF'].str.replace('Average', 'C',regex=False)
w_eff['WINDOWS_ENERGY_EFF'] = w_eff['WINDOWS_ENERGY_EFF'].str.replace('Poor', 'D',regex=False)
w_eff['WINDOWS_ENERGY_EFF'] = w_eff['WINDOWS_ENERGY_EFF'].str.replace('Very D', 'E',regex=False)
w_eff.head()


# Average Good Poor Very Good 
wa_eff = data[['LMK_KEY','WALLS_ENERGY_EFF']]
wa_eff.head()
wa_eff.isnull().sum()
wa_eff.dropna(inplace = True)
wa_eff.isnull().sum()
wa_eff.value_counts()
wa_eff['WALLS_ENERGY_EFF'] = wa_eff['WALLS_ENERGY_EFF'].str.replace('Very Good', 'A',regex=False)
wa_eff['WALLS_ENERGY_EFF'] = wa_eff['WALLS_ENERGY_EFF'].str.replace('Good', 'B',regex=False)
wa_eff['WALLS_ENERGY_EFF'] = wa_eff['WALLS_ENERGY_EFF'].str.replace('Average', 'C',regex=False)
wa_eff['WALLS_ENERGY_EFF'] = wa_eff['WALLS_ENERGY_EFF'].str.replace('Poor', 'D',regex=False)
wa_eff['WALLS_ENERGY_EFF'] = wa_eff['WALLS_ENERGY_EFF'].str.replace('Very D', 'E',regex=False)
wa_eff.value_counts()

wa_eff.head()

hot_eff = data[['LMK_KEY','HOT_WATER_ENERGY_EFF']]
hot_eff.head()
hot_eff.isnull().sum()
hot_eff.dropna(inplace = True)
hot_eff.isnull().sum()
hot_eff.value_counts()
hot_eff['HOT_WATER_ENERGY_EFF'] = hot_eff['HOT_WATER_ENERGY_EFF'].str.replace('Very Good', 'A',regex=False)
hot_eff['HOT_WATER_ENERGY_EFF'] = hot_eff['HOT_WATER_ENERGY_EFF'].str.replace('Good', 'B',regex=False)
hot_eff['HOT_WATER_ENERGY_EFF'] = hot_eff['HOT_WATER_ENERGY_EFF'].str.replace('Average', 'C',regex=False)
hot_eff['HOT_WATER_ENERGY_EFF'] = hot_eff['HOT_WATER_ENERGY_EFF'].str.replace('Poor', 'D',regex=False)
hot_eff['HOT_WATER_ENERGY_EFF'] = hot_eff['HOT_WATER_ENERGY_EFF'].str.replace('Very D', 'E',regex=False)

heat_eff = data[['LMK_KEY','MAINHEAT_ENERGY_EFF']]
heat_eff.head()
heat_eff.isnull().sum()
heat_eff.dropna(inplace = True)
heat_eff.isnull().sum()
heat_eff['MAINHEAT_ENERGY_EFF'].value_counts()
heat_eff['MAINHEAT_ENERGY_EFF'] = heat_eff['MAINHEAT_ENERGY_EFF'].str.replace('Very Good', 'A',regex=False)
heat_eff['MAINHEAT_ENERGY_EFF'] = heat_eff['MAINHEAT_ENERGY_EFF'].str.replace('Good', 'B',regex=False)
heat_eff['MAINHEAT_ENERGY_EFF'] = heat_eff['MAINHEAT_ENERGY_EFF'].str.replace('Average', 'C',regex=False)
heat_eff['MAINHEAT_ENERGY_EFF'] = heat_eff['MAINHEAT_ENERGY_EFF'].str.replace('Poor', 'D',regex=False)
heat_eff['MAINHEAT_ENERGY_EFF'] = heat_eff['MAINHEAT_ENERGY_EFF'].str.replace('Very D', 'E',regex=False)

lit_eff = data[['LMK_KEY','LIGHTING_ENERGY_EFF']]
lit_eff.head()
lit_eff.isnull().sum()
lit_eff.dropna(inplace = True)
lit_eff.isnull().sum()
lit_eff['LIGHTING_ENERGY_EFF'].value_counts()
lit_eff['LIGHTING_ENERGY_EFF'] = lit_eff['LIGHTING_ENERGY_EFF'].str.replace('Very Good', 'A',regex=False)
lit_eff['LIGHTING_ENERGY_EFF'] = lit_eff['LIGHTING_ENERGY_EFF'].str.replace('Good', 'B',regex=False)
lit_eff['LIGHTING_ENERGY_EFF'] = lit_eff['LIGHTING_ENERGY_EFF'].str.replace('Average', 'C',regex=False)
lit_eff['LIGHTING_ENERGY_EFF'] = lit_eff['LIGHTING_ENERGY_EFF'].str.replace('Poor', 'D',regex=False)
lit_eff['LIGHTING_ENERGY_EFF'] = lit_eff['LIGHTING_ENERGY_EFF'].str.replace('Very D', 'E',regex=False)

# merge
a_eff = pd.merge(w_eff, wa_eff, on ='LMK_KEY', how = 'inner')
a_eff = pd.merge( hot_eff , a_eff, on ='LMK_KEY', how = 'inner')
a_eff = pd.merge( heat_eff , a_eff, on ='LMK_KEY', how = 'inner')
a_eff = pd.merge( lit_eff , a_eff, on ='LMK_KEY', how = 'inner')

efff = data[['LMK_KEY','CURRENT_ENERGY_EFFICIENCY']]
a_eff = pd.merge(efff, a_eff, on = 'LMK_KEY', how = 'inner')
a_eff.head()
def dummy_data(data, columns):
        for column in columns:
                data = pd.concat([data, pd.get_dummies(data[column], prefix = column)], axis= 1)
                data = data.drop(column, axis=1)
        return data
d_columns = ['HOT_WATER_ENERGY_EFF','MAINHEAT_ENERGY_EFF' ,'LIGHTING_ENERGY_EFF']
a_eff_d = dummy_data(a_eff, d_columns)
a_eff_d.keys()
### K mean for other services 

kmean_df_eff = a_eff_d[['LMK_KEY', 'CURRENT_ENERGY_EFFICIENCY', 'WINDOWS_ENERGY_EFF',
       'WALLS_ENERGY_EFF', 'HOT_WATER_ENERGY_EFF_A', 'HOT_WATER_ENERGY_EFF_B',
       'HOT_WATER_ENERGY_EFF_C', 'HOT_WATER_ENERGY_EFF_D',
       'HOT_WATER_ENERGY_EFF_E', 'MAINHEAT_ENERGY_EFF_A',
       'MAINHEAT_ENERGY_EFF_B', 'MAINHEAT_ENERGY_EFF_C',
       'MAINHEAT_ENERGY_EFF_D', 'MAINHEAT_ENERGY_EFF_E',
       'LIGHTING_ENERGY_EFF_A', 'LIGHTING_ENERGY_EFF_B',
       'LIGHTING_ENERGY_EFF_C', 'LIGHTING_ENERGY_EFF_D',
       'LIGHTING_ENERGY_EFF_E']]
columns_eff = ['HOT_WATER_ENERGY_EFF_A', 'HOT_WATER_ENERGY_EFF_B',
       'HOT_WATER_ENERGY_EFF_C', 'HOT_WATER_ENERGY_EFF_D',
       'HOT_WATER_ENERGY_EFF_E', 'MAINHEAT_ENERGY_EFF_A',
       'MAINHEAT_ENERGY_EFF_B', 'MAINHEAT_ENERGY_EFF_C',
       'MAINHEAT_ENERGY_EFF_D', 'MAINHEAT_ENERGY_EFF_E',
       'LIGHTING_ENERGY_EFF_A', 'LIGHTING_ENERGY_EFF_B',
       'LIGHTING_ENERGY_EFF_C', 'LIGHTING_ENERGY_EFF_D',
       'LIGHTING_ENERGY_EFF_E']
labels_eff = np.array(a_eff_d['LMK_KEY'])
labels_eff
kmean_df_eff = kmean_df_eff.drop(['CURRENT_ENERGY_EFFICIENCY', 'WINDOWS_ENERGY_EFF',
       'WALLS_ENERGY_EFF','LMK_KEY'], axis = 1)
kmean_df_eff.head()
stdsc_eff = StandardScaler()
kmean_df_eff = pd.DataFrame(stdsc_eff.fit_transform(kmean_df_eff))
kmean_df_eff.index = labels_eff
kmean_df_eff.columns = columns_eff
kmean_df_eff.head()
kmean_df_eff.keys()
kvalue_eff = kmean_df_eff.values

km_eff = KMeans(n_clusters = 8,\
        init = 'k-means++',\
        n_init = 20,\
        max_iter = 300,\
        random_state=42)
y_km_eff = km_eff.fit_predict(kmean_df_eff)
y_km_eff

from yellowbrick.cluster import KElbowVisualizer

model_eff = KMeans()
visualizer_eff = KElbowVisualizer(model_eff, k=(1,20))
visualizer_eff.fit(kmean_df_eff)
plt.title('Elbow method for optimal value of K in K-means')
plt.show()

kmean_df_eff.head()

K_HOT_WATER = kmean_df_eff[['HOT_WATER_ENERGY_EFF_A','HOT_WATER_ENERGY_EFF_B',\
        'HOT_WATER_ENERGY_EFF_C','HOT_WATER_ENERGY_EFF_D','HOT_WATER_ENERGY_EFF_E']]
K_HOT_WATER.head()
K_HOT_WATER = K_HOT_WATER.sum(axis=1)
K_HOT_WATER.sort_index()
K_HOT_WATER.index



kmean_df_eff['cluster'] = y_km_eff
kmean_df1_eff = kmean_df_eff[kmean_df_eff.cluster==0]
kmean_df2_eff = kmean_df_eff[kmean_df_eff.cluster==1]
kmean_df3_eff = kmean_df_eff[kmean_df_eff.cluster==2]
kmean_df4_eff = kmean_df_eff[kmean_df_eff.cluster==3]
kmean_df5_eff = kmean_df_eff[kmean_df_eff.cluster==4]
kmean_df6_eff = kmean_df_eff[kmean_df_eff.cluster==5]
kmean_df7_eff = kmean_df_eff[kmean_df_eff.cluster==6]
kmean_df8_eff = kmean_df_eff[kmean_df_eff.cluster==7]
kmean_df9_eff = kmean_df_eff[kmean_df_eff.cluster==8]
kmean_df7_eff.max()



kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)
kplot.plot3D(xline, yline, zline, 'red')
# # Data for three-dimensional scattered points
kplot.scatter3D(kmean_df1_eff.HOT_WATER_ENERGY_EFF_A, kmean_df1_eff.MAINHEAT_ENERGY_EFF_A, kmean_df1_eff.LIGHTING_ENERGY_EFF_A, c='darkred', label = 'Cluster 1',marker= "1",s = 100)
kplot.scatter3D(kmean_df1_eff.HOT_WATER_ENERGY_EFF_B, kmean_df1_eff.MAINHEAT_ENERGY_EFF_B, kmean_df1_eff.LIGHTING_ENERGY_EFF_B,c='darkred',marker= "1",s = 100)
kplot.scatter3D(kmean_df1_eff.HOT_WATER_ENERGY_EFF_C, kmean_df1_eff.MAINHEAT_ENERGY_EFF_C, kmean_df1_eff.LIGHTING_ENERGY_EFF_C,c='darkred',marker= "1",s = 100)
kplot.scatter3D(kmean_df1_eff.HOT_WATER_ENERGY_EFF_D, kmean_df1_eff.MAINHEAT_ENERGY_EFF_D, kmean_df1_eff.LIGHTING_ENERGY_EFF_D,c='darkred',marker= "1",s = 100)
kplot.scatter3D(kmean_df1_eff.HOT_WATER_ENERGY_EFF_E, kmean_df1_eff.MAINHEAT_ENERGY_EFF_E, kmean_df1_eff.LIGHTING_ENERGY_EFF_E,c='darkred',marker= "1",s = 100)

kplot.scatter3D(kmean_df2_eff.HOT_WATER_ENERGY_EFF_A, kmean_df2_eff.MAINHEAT_ENERGY_EFF_A, kmean_df2_eff.LIGHTING_ENERGY_EFF_A,c='gold', label = 'Cluster 2',marker= "2",s = 100)
kplot.scatter3D(kmean_df2_eff.HOT_WATER_ENERGY_EFF_B, kmean_df2_eff.MAINHEAT_ENERGY_EFF_B, kmean_df2_eff.LIGHTING_ENERGY_EFF_B,c='gold', marker= "2",s = 100)
kplot.scatter3D(kmean_df2_eff.HOT_WATER_ENERGY_EFF_C, kmean_df2_eff.MAINHEAT_ENERGY_EFF_C, kmean_df2_eff.LIGHTING_ENERGY_EFF_C,c='gold', marker= "2",s = 100)
kplot.scatter3D(kmean_df2_eff.HOT_WATER_ENERGY_EFF_D, kmean_df2_eff.MAINHEAT_ENERGY_EFF_D, kmean_df2_eff.LIGHTING_ENERGY_EFF_D,c='gold', marker= "2",s = 100)
kplot.scatter3D(kmean_df2_eff.HOT_WATER_ENERGY_EFF_E, kmean_df2_eff.MAINHEAT_ENERGY_EFF_E, kmean_df2_eff.LIGHTING_ENERGY_EFF_E,c='gold', marker= "2",s = 100)

kplot.scatter3D(kmean_df3_eff.HOT_WATER_ENERGY_EFF_A, kmean_df3_eff.MAINHEAT_ENERGY_EFF_A, kmean_df3_eff.LIGHTING_ENERGY_EFF_A,c='teal', label = 'Cluster 3',marker= "3",s = 100)
kplot.scatter3D(kmean_df3_eff.HOT_WATER_ENERGY_EFF_B, kmean_df3_eff.MAINHEAT_ENERGY_EFF_B, kmean_df3_eff.LIGHTING_ENERGY_EFF_B,c='teal', marker= "3",s = 100)
kplot.scatter3D(kmean_df3_eff.HOT_WATER_ENERGY_EFF_C, kmean_df3_eff.MAINHEAT_ENERGY_EFF_C, kmean_df3_eff.LIGHTING_ENERGY_EFF_C,c='teal', marker= "3",s = 100)
kplot.scatter3D(kmean_df3_eff.HOT_WATER_ENERGY_EFF_D, kmean_df3_eff.MAINHEAT_ENERGY_EFF_D, kmean_df3_eff.LIGHTING_ENERGY_EFF_D,c='teal',marker= "3",s = 100)
kplot.scatter3D(kmean_df3_eff.HOT_WATER_ENERGY_EFF_E, kmean_df3_eff.MAINHEAT_ENERGY_EFF_E, kmean_df3_eff.LIGHTING_ENERGY_EFF_E,c='teal', marker= "3",s = 100)

kplot.scatter3D(kmean_df4_eff.HOT_WATER_ENERGY_EFF_A, kmean_df4_eff.MAINHEAT_ENERGY_EFF_A, kmean_df4_eff.LIGHTING_ENERGY_EFF_A,c='deeppink', label = 'Cluster 4',marker= "4",s = 100)
kplot.scatter3D(kmean_df4_eff.HOT_WATER_ENERGY_EFF_B, kmean_df4_eff.MAINHEAT_ENERGY_EFF_B, kmean_df4_eff.LIGHTING_ENERGY_EFF_B,c='deeppink', marker= "4",s = 100)
kplot.scatter3D(kmean_df4_eff.HOT_WATER_ENERGY_EFF_C, kmean_df4_eff.MAINHEAT_ENERGY_EFF_C, kmean_df4_eff.LIGHTING_ENERGY_EFF_C,c='deeppink', marker= "4",s = 100)
kplot.scatter3D(kmean_df4_eff.HOT_WATER_ENERGY_EFF_D, kmean_df4_eff.MAINHEAT_ENERGY_EFF_D, kmean_df4_eff.LIGHTING_ENERGY_EFF_D,c='deeppink', marker= "4",s = 100)
kplot.scatter3D(kmean_df4_eff.HOT_WATER_ENERGY_EFF_E, kmean_df4_eff.MAINHEAT_ENERGY_EFF_E, kmean_df4_eff.LIGHTING_ENERGY_EFF_E,c='deeppink', marker= "4",s = 100)

kplot.scatter3D(kmean_df5_eff.HOT_WATER_ENERGY_EFF_A, kmean_df5_eff.MAINHEAT_ENERGY_EFF_A, kmean_df5_eff.LIGHTING_ENERGY_EFF_A,c='orange', label = 'Cluster 5',marker= "v",s = 50)
kplot.scatter3D(kmean_df5_eff.HOT_WATER_ENERGY_EFF_B, kmean_df5_eff.MAINHEAT_ENERGY_EFF_B, kmean_df5_eff.LIGHTING_ENERGY_EFF_B,c='orange', marker= "v",s = 50)
kplot.scatter3D(kmean_df5_eff.HOT_WATER_ENERGY_EFF_C, kmean_df5_eff.MAINHEAT_ENERGY_EFF_C, kmean_df5_eff.LIGHTING_ENERGY_EFF_C,c='orange', marker= "v",s = 50)
kplot.scatter3D(kmean_df5_eff.HOT_WATER_ENERGY_EFF_D, kmean_df5_eff.MAINHEAT_ENERGY_EFF_D, kmean_df5_eff.LIGHTING_ENERGY_EFF_D,c='orange', marker= "v",s = 50)
kplot.scatter3D(kmean_df5_eff.HOT_WATER_ENERGY_EFF_E, kmean_df5_eff.MAINHEAT_ENERGY_EFF_E, kmean_df5_eff.LIGHTING_ENERGY_EFF_E,c='orange', marker= "v",s = 50)

kplot.scatter3D(kmean_df6_eff.HOT_WATER_ENERGY_EFF_A, kmean_df6_eff.MAINHEAT_ENERGY_EFF_A, kmean_df6_eff.LIGHTING_ENERGY_EFF_A,c='lightcoral', label = 'Cluster 6',marker= "^",s = 50)
kplot.scatter3D(kmean_df6_eff.HOT_WATER_ENERGY_EFF_B, kmean_df6_eff.MAINHEAT_ENERGY_EFF_B, kmean_df6_eff.LIGHTING_ENERGY_EFF_B,c='lightcoral',marker= "^",s = 50)
kplot.scatter3D(kmean_df6_eff.HOT_WATER_ENERGY_EFF_C, kmean_df6_eff.MAINHEAT_ENERGY_EFF_C, kmean_df6_eff.LIGHTING_ENERGY_EFF_C,c='lightcoral',marker= "^",s = 50)
kplot.scatter3D(kmean_df6_eff.HOT_WATER_ENERGY_EFF_D, kmean_df6_eff.MAINHEAT_ENERGY_EFF_D, kmean_df6_eff.LIGHTING_ENERGY_EFF_D,c='lightcoral',marker= "^",s = 50)
kplot.scatter3D(kmean_df6_eff.HOT_WATER_ENERGY_EFF_E, kmean_df6_eff.MAINHEAT_ENERGY_EFF_E, kmean_df6_eff.LIGHTING_ENERGY_EFF_E,c='lightcoral',marker= "^",s = 50)

kplot.scatter3D(kmean_df7_eff.HOT_WATER_ENERGY_EFF_A, kmean_df7_eff.MAINHEAT_ENERGY_EFF_A, kmean_df7_eff.LIGHTING_ENERGY_EFF_A,c='crimson', label = 'Cluster 7',marker= "<",s = 50)
kplot.scatter3D(kmean_df7_eff.HOT_WATER_ENERGY_EFF_B, kmean_df7_eff.MAINHEAT_ENERGY_EFF_B, kmean_df7_eff.LIGHTING_ENERGY_EFF_B,c='crimson',marker= "<",s = 50)
kplot.scatter3D(kmean_df7_eff.HOT_WATER_ENERGY_EFF_C, kmean_df7_eff.MAINHEAT_ENERGY_EFF_C, kmean_df7_eff.LIGHTING_ENERGY_EFF_C,c='crimson',marker= "<",s = 50)
kplot.scatter3D(kmean_df7_eff.HOT_WATER_ENERGY_EFF_D, kmean_df7_eff.MAINHEAT_ENERGY_EFF_D, kmean_df7_eff.LIGHTING_ENERGY_EFF_D,c='crimson',marker= "<",s = 50)
kplot.scatter3D(kmean_df7_eff.HOT_WATER_ENERGY_EFF_E, kmean_df7_eff.MAINHEAT_ENERGY_EFF_E, kmean_df7_eff.LIGHTING_ENERGY_EFF_E,c='crimson',marker= "<",s = 50)

kplot.scatter3D(kmean_df8_eff.HOT_WATER_ENERGY_EFF_A, kmean_df8_eff.MAINHEAT_ENERGY_EFF_A, kmean_df8_eff.LIGHTING_ENERGY_EFF_A,c='blueviolet', label = 'Cluster 8',marker= ">",s = 50)
kplot.scatter3D(kmean_df8_eff.HOT_WATER_ENERGY_EFF_B, kmean_df8_eff.MAINHEAT_ENERGY_EFF_B, kmean_df8_eff.LIGHTING_ENERGY_EFF_B,c='blueviolet', marker= ">",s = 50)
kplot.scatter3D(kmean_df8_eff.HOT_WATER_ENERGY_EFF_C, kmean_df8_eff.MAINHEAT_ENERGY_EFF_C, kmean_df8_eff.LIGHTING_ENERGY_EFF_C,c='blueviolet', marker= ">",s = 50)
kplot.scatter3D(kmean_df8_eff.HOT_WATER_ENERGY_EFF_D, kmean_df8_eff.MAINHEAT_ENERGY_EFF_D, kmean_df8_eff.LIGHTING_ENERGY_EFF_D,c='blueviolet', marker= ">",s = 50)
kplot.scatter3D(kmean_df8_eff.HOT_WATER_ENERGY_EFF_E, kmean_df8_eff.MAINHEAT_ENERGY_EFF_E, kmean_df8_eff.LIGHTING_ENERGY_EFF_E,c='blueviolet', marker= ">",s = 50)


plt.scatter(km_eff.cluster_centers_[:,0], km_eff.cluster_centers_[:,1], color = 'dodgerblue', s = 100, marker= "*", label = 'Centroids')
plt.legend()
plt.grid(False)
plt.title("K-means (8 Clusters)")
plt.show()

from sklearn.metrics import silhouette_score
score_eff = silhouette_score(kvalue_eff,y_km_eff)
print(score_eff)

cluster_labels = np.unique(y_km_eff)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(kmean_df_eff, y_km_eff, metric = 'euclidean')
y_ax_lower, y_ax_upper = 0,0
yticks = []
for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km_eff == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height = 1.0,
        edgecolor = 'none', color = color)

        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(c_silhouette_vals)

plt.axvline(silhouette_avg, color = 'red', linestyle= '--')
plt.yticks(yticks, cluster_labels +1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()




cluster4 = pd.DataFrame(y_km)
cluster4.head()
cluster4.index = labels
cluster4.columns = ['cluster4']


print(cluster4.index.dtype)
print(y_km.dtype)
cluster5 = pd.DataFrame(y_km)
cluster5.index = labels
cluster5.columns = ['cluster5']
print(cluster5)

print(cluster3)

sys.stdout = open('cluster3.txt','w')
print(cluster3)
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

y_kmeans_eff = km_eff.fit_predict(kvalue_eff)
y_kmeans_eff

plt.figure(figsize = (8, 8))
plt.scatter(kvalue_eff[y_kmeans == 0, 0], kvalue[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(kvalue_eff[y_kmeans == 1, 0], kvalue[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(kvalue_eff[y_kmeans == 2, 0], kvalue[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(kvalue[y_kmeans == 3, 0], kvalue[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.legend() 
plt.show()

a_eff.keys()


a_eff1 = pd.get_dummies(a_eff.HOT_WATER_ENERGY_EFF)
a_eff2 = pd.get_dummies(a_eff.WINDOWS_ENERGY_EFF)
a_eff4 = pd.get_dummies(a_eff.LIGHTING_ENERGY_EFF)
a_eff3 = pd.get_dummies(a_eff.WALLS_ENERGY_EFF)
a_eff5 = pd.get_dummies(a_eff.MAINHEAT_ENERGY_EFF)


a_eff1['intercept'] = 1 
sm.OLS(a_eff['CURRENT_ENERGY_EFFICIENCY'], a_eff1[['intercept', 'B', 'C', 'D', 'E']]).fit().summary()
sys.stdout = open('HOT_WATER_ENERGY_EFF.txt','w')
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

a_eff2['intercept'] = 1 
sm.OLS(a_eff['CURRENT_ENERGY_EFFICIENCY'], a_eff2[['intercept', 'B', 'C', 'D', 'E']]).fit().summary()
sys.stdout = open('WALLS_ENERGY_EFF.txt','w')
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

a_eff3['intercept'] = 1 
sm.OLS(a_eff['CURRENT_ENERGY_EFFICIENCY'], a_eff3[['intercept', 'B', 'C', 'D', 'E']]).fit().summary()


a_eff4['intercept'] = 1 
sm.OLS(a_eff['CURRENT_ENERGY_EFFICIENCY'], a_eff4[['intercept', 'B', 'C', 'D', 'E']]).fit().summary()
sys.stdout = open('a_eff.LIGHTING_ENERGY_EFF.txt','w')
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

a_eff5['intercept'] = 1 
sm.OLS(a_eff['CURRENT_ENERGY_EFFICIENCY'], a_eff5[['intercept', 'B', 'C', 'D', 'E']]).fit().summary()
sys.stdout = open('MAINHEAT_ENERGY_EFF.txt','w')
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

### K mean for the cost 
kmean_df = data[['HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT', 'HOT_WATER_COST_CURRENT','LMK_KEY']]
columns = ['HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT', 'HOT_WATER_COST_CURRENT']
labels = np.array(data['LMK_KEY'])
labels
kmean_df = kmean_df.drop(['LMK_KEY'], axis = 1)
kmean_df.head()
stdsc = StandardScaler()
kmean_df = pd.DataFrame(stdsc.fit_transform(kmean_df))
kmean_df.index = labels
kmean_df.columns = columns
kmean_df.head()
kvalue = kmean_df.values
kmean_df['HOT_WATER_COST_CURRENT'].max()
kmean_df['HOT_WATER_COST_CURRENT'].min()
kmean_df['LIGHTING_COST_CURRENT'].max()
kmean_df['LIGHTING_COST_CURRENT'].min()
kmean_df['HEATING_COST_CURRENT'].max()
kmean_df['HEATING_COST_CURRENT'].min()


km = KMeans(n_clusters = 4,\
        init = 'k-means++',\
        n_init = 10,\
        max_iter = 300,\
        random_state=1)
y_km = km.fit_predict(kmean_df)
y_km
km.cluster_centers_

kmean_df['cluster'] = y_km
kmean_df1 = kmean_df[kmean_df.cluster==0]
kmean_df2 = kmean_df[kmean_df.cluster==1]
kmean_df3 = kmean_df[kmean_df.cluster==2]
kmean_df4 = kmean_df[kmean_df.cluster==3]


kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)
kplot.plot3D(xline, yline, zline, 'red')

# Data for three-dimensional scattered points
kplot.scatter3D(kmean_df1.HEATING_COST_CURRENT, kmean_df1.LIGHTING_COST_CURRENT, kmean_df1.HOT_WATER_COST_CURRENT, c='darkred', label = 'Cluster 1',marker= "1",s = 100)
kplot.scatter3D(kmean_df2.HEATING_COST_CURRENT,kmean_df2.LIGHTING_COST_CURRENT,kmean_df2.HOT_WATER_COST_CURRENT,c ='oldlace', label = 'Cluster 2',marker= "2",s = 100)
kplot.scatter3D(kmean_df3.HEATING_COST_CURRENT,kmean_df3.LIGHTING_COST_CURRENT,kmean_df3.HOT_WATER_COST_CURRENT,c ='teal', label = 'Cluster 3',marker= "3",s = 100)
kplot.scatter3D(kmean_df4.HEATING_COST_CURRENT,kmean_df4.LIGHTING_COST_CURRENT,kmean_df4.HOT_WATER_COST_CURRENT,c ='deeppink', label = 'Cluster 4',marker= "4",s = 100)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'dodgerblue', s = 100, marker= "*", label = 'Centroids')
plt.legend()
plt.grid(False)
plt.title("K-means (4 Clusters)")
plt.show()

from sklearn.metrics import silhouette_score
score = silhouette_score(kvalue,y_km)
print(score)


cluster4 = pd.DataFrame(y_km)
cluster4.head()
cluster4.index = labels
cluster4.columns = ['cluster4']


print(cluster4.index.dtype)
print(y_km.dtype)
cluster5 = pd.DataFrame(y_km)
cluster5.index = labels
cluster5.columns = ['cluster5']
print(cluster5)

print(cluster3)

sys.stdout = open('cluster3.txt','w')
print(cluster3)
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

y_kmeans_eff = km_eff.fit_predict(kvalue_eff)
y_kmeans_eff

plt.figure(figsize = (8, 8))
plt.scatter(kvalue_eff[y_kmeans_eff == 0, 0], kvalue_eff[y_kmeans_eff == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(kvalue_eff[y_kmeans_eff == 1, 0], kvalue_eff[y_kmeans_eff == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(kvalue_eff[y_kmeans_eff == 2, 0], kvalue_eff[y_kmeans_eff == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(kvalue_eff[y_kmeans_eff == 3, 0], kvalue_eff[y_kmeans_eff == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
plt.scatter(km_eff.cluster_centers_[:, 0], km_eff.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.legend() 
plt.show()



# elbow
distortions = []
for i in range(1,10):
        km = KMeans(n_clusters=i,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=0)
        km.fit(kmean_df)
        distortions.append(km.inertia_)
plt.plot(range(1,10), distortions, marker = 'o')
plt.annotate('Elbow Point', xy=(3.80,97800), xytext=(2.20,50000), arrowprops=dict(arrowstyle='->'))
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for optimal value of k in KMeans')
plt.tight_layout()
plt.show()

clusterDF['meanshift_label'] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ['o','s','^','x','*']

for label in unique_labels:
    label_cluster = clusterDF[clusterDF['meanshift_label']==label]
    center_x_y = centers[label]
    # 군집별로 다른 마커로 산점도 적용
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'],
                edgecolor='k', marker=markers[label])
    
    # 군집별 중심 표현
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='gray',
                alpha=0.9, marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k',
                marker='$%d$' % label)
    
plt.show()
from yellowbrick.cluster import KElbowVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
visualizer.fit(kmean_df)
plt.show()

# K means -silhouette_avg
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(kmean_df, y_km, metric = 'euclidean')
y_ax_lower, y_ax_upper = 0,0
yticks = []
for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height = 1.0,
        edgecolor = 'none', color = color)

        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(c_silhouette_vals)

plt.axvline(silhouette_avg, color = 'red', linestyle= '--')
plt.yticks(yticks, cluster_labels +1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.show()

# efficiency
effff = pd.merge(w_eff, wa_eff, on ='LMK_KEY', how = 'inner')
effff = pd.merge( hot_eff , effff, on ='LMK_KEY', how = 'inner')
effff = pd.merge( efff , effff, on ='LMK_KEY', how = 'inner')
effff1 = data[['CURRENT_ENERGY_RATING','LMK_KEY']]
effff = pd.merge( effff1 , effff, on ='LMK_KEY', how = 'inner')
effff.head()
effff.keys()
effff.shape
X = effff.drop(['LMK_KEY','CURRENT_ENERGY_RATING'], axis= 1)
X.head()
y = effff['CURRENT_ENERGY_RATING']
y.head()
class_le = LabelEncoder()
y = class_le.fit_transform(y)
y

baseline_accuracy = y.max()/y.sum()
baseline_accuracy
#  1.1936119796627617e-05 accuracy

X.keys()
X = pd.get_dummies(X[['CURRENT_ENERGY_EFFICIENCY', 'HOT_WATER_ENERGY_EFF',
       'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF']],\
        columns = ['HOT_WATER_ENERGY_EFF', 'WINDOWS_ENERGY_EFF', 'WALLS_ENERGY_EFF'],
        drop_first = True)
X.head()

X_train, X_test, y_train, y_test = \
        train_test_split(X,y,
        test_size =0.3,
        random_state=1,
        stratify=y)

stdsc = StandardScaler()
X_train.iloc[:,[0]] = stdsc.fit_transform(X_train.iloc[:,[0]])
X_test.iloc[:,[0]] = stdsc.transform(X_test.iloc[:,[0]])
X_test.head()

svm = SVC(kernel = 'rbf',\
        random_state =1,
        gamma =0.2,
        C =1.0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
y_pred

confmat = pd.DataFrame(confusion_matrix(y_test, y_pred),\
        index = ['True[0]', 'True[1]','True[2]','True[3]','True[4]','True[5]','True[6]'],
        columns=['Predict[0]', 'Predict[1]','Predict[2]','Predict[3]','Predict[4]','Predict[5]','Predict[6]'])
confmat

sys.stdout = open('Classification Report_svm.txt','w')
print(confmat)
sys.stdout.close()
sys.stdout = open("/dev/stdout", "w")

print('Classification Report')
print(classification_report(y_test, y_pred))

print('The number of samples with a wrong prediction : %d' %(y_test != y_pred).sum())
print('Accuracy Score: %.3f' %accuracy_score(y_test,y_pred))

pipe_svm = make_pipeline(SVC(random_state =1))
pipe_svm.get_params().keys()

train_sizes, train_scores, test_score = \
        learning_curve(estimator =pipe_svm,
        X= X_train,
        y=y_train,
        train_sizes=np.linspace(0.1,1.0,10),
        cv=10,
        n_jobs =1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis =1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,\
        color = 'blue', marker = 'o',
        markersize =5, lable = 'training accuracy')
plt.fill_between(train_sizes,\
        train_mean + test_std,
        test_mean - test_std,
        alpha = 0.15, color ='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,0.9])
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(X.iloc[:,0], X.iloc[:,1], marker='o', s=100, alpha=0.5)
plt.grid()
plt.show()


#################
data.keys()
kmean_df = data[['HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT', 'HOT_WATER_COST_CURRENT','LMK_KEY']]
columns = ['HEATING_COST_CURRENT', 'LIGHTING_COST_CURRENT', 'HOT_WATER_COST_CURRENT']
labels = np.array(data['LMK_KEY'])
labels
kmean_df = kmean_df.drop(['LMK_KEY'], axis = 1)
kmean_df.head()
stdsc = StandardScaler()
kmean_df = pd.DataFrame(stdsc.fit_transform(kmean_df))
kmean_df.index = labels
kmean_df.columns = columns
kmean_df.head()
kvalue = kmean_df.values
kmean_df['HOT_WATER_COST_CURRENT'].max()
kmean_df['HOT_WATER_COST_CURRENT'].min()
kmean_df['LIGHTING_COST_CURRENT'].max()
kmean_df['LIGHTING_COST_CURRENT'].min()
kmean_df['HEATING_COST_CURRENT'].max()
kmean_df['HEATING_COST_CURRENT'].min()


km = KMeans(n_clusters = 4,\
        init = 'k-means++',\
        n_init = 10,\
        max_iter = 300,\
        random_state=1)
y_km = km.fit_predict(kmean_df)
y_km
km.cluster_centers_