# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:08:17 2022

@author: admin
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn import neighbors
from math import sqrt
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score


#################################################################
#####################preprocessing##############################
#################################################################

ds=pd.read_csv("russia_war.csv")
#check missing values 
ds.isna().sum()
#change to date object
ds['dateOlose'] = pd.to_datetime(ds['dateOlose'])
#transform  to unifide unites for all equipments

ds_equ=ds[["aircaft","helicopter","tank","APC","field_artillery","MRL","military_auto"
   ,"fuel_tank","drone","naval_ship","anti_aircarft_warfare","special_equipments"]]
################################transformation######################################
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

ds_trans_equ=ds[["aircaft","helicopter","tank","APC","field_artillery","MRL","military_auto"
   ,"fuel_tank","drone","naval_ship","anti_aircarft_warfare","special_equipments"]]
ds1_scale = scaler.fit_transform(ds_trans_equ)

ds_trans_equ = pd.DataFrame(ds1_scale , index=ds_trans_equ.index, columns=ds_trans_equ.columns )

#################################################################
#####################Data Analytics##############################
#################################################################
######## most loses equipments########
######################################


sum_each_equ=ds_trans_equ.sum().sort_values()
ds_sum_each_equ=pd.DataFrame()
ds_sum_each_equ['sum_each_equiment']=pd.DataFrame(sum_each_equ)
ds_sum_each_equ.reset_index(inplace=True)

x=ds_sum_each_equ.iloc[:,0]
y=ds_sum_each_equ.iloc[:,-1]
fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1] )
ax.bar(x,y,color ='red')
plt.xticks(rotation=50)
plt.title('most losses equipments')
plt.xlabel('Equipment Name')
plt.show()



#######################################
## equpments loses per day##
#######################################
sum_equ_day=ds_trans_equ.sum(axis=1)
ds_equ_day=pd.DataFrame()
ds_equ_day['total of the losses']=pd.DataFrame(sum_equ_day)
ds_equ_day["Day Number"]=pd.DataFrame(ds['num_of_day'])
x=ds_equ_day.iloc[:,0]
y=ds_equ_day.iloc[:,-1]
plt.plot(x,y)
plt.title('daily losses equipments')
plt.xlabel('Equipment losses')
plt.ylabel('Days')
plt.show()
########################################
#######################################
#total of equpments loses per week##
#######################################
ds_week=pd.DataFrame()
ds_week['total equpments']=pd.DataFrame(sum_equ_day)
ds_week['dateOlose']=pd.DataFrame(ds['dateOlose'])
ds_week['Week_Number'] = ds_week['dateOlose'].dt.week
ds_week.drop(['dateOlose'],axis=1)
ds_total_week=pd.DataFrame(ds_week.groupby('Week_Number')['total equpments'].sum())
ds_total_week.reset_index(inplace=True)
x=ds_total_week.iloc[:,0]
y=ds_total_week.iloc[:,-1]
x_sk=x.skew()
y_sk=y.skew()
fig = plt.figure()
ax = fig.add_axes([0,0,1,1] )
ax.bar(x,y,color ='blue')
plt.title('total loses equipments per week')
plt.xlabel('Week Number')
plt.show()

####################################################################
##################relation between presioners and equipents#########
####################################################################

#######################################################################
#############################
sum_equ_day=ds_equ.sum(axis=1)
total_prosinor=ds[['personnel','POW']]
total_prosinor=total_prosinor.sum(axis=1)
total_pre_equ=pd.DataFrame()
total_pre_equ['total_presonors']=pd.DataFrame(total_prosinor)
total_pre_equ['total_equ']=pd.DataFrame(sum_equ_day)
x=total_pre_equ.iloc[:,0]
y=total_pre_equ.iloc[:,-1]
plt.plot(x,y)
plt.title('daily presinors and equ')
plt.xlabel('Equipment losses')
plt.ylabel('POW')
plt.show()
ax=sb.kdeplot(total_pre_equ['total_presonors'], bw_adjust=2, cut=0.2)
plt.show()
sk_pres=x.skew()
sk_equ=y.skew()

######################################################################


# #########################################
# ##########corrlation#####################


corrtable=total_pre_equ.corr()
corrtable.reset_index(inplace=True)


#######################################

x=total_pre_equ.iloc[:,-1]
x = x.values.reshape(-1,1)

y=total_pre_equ.iloc[:,0]
y = y.values.reshape(-1,1)



from sklearn.model_selection import train_test_split 

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.30 , random_state=0 )

#train 
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train , y_train)

# q4 predict the traing set 
y_predict= regressor.predict(x_test)
r2 = r2_score(y_test, y_predict)


print('\n \n \n')
print('The Performance of Linear Regression:',r2,'%')

plt.scatter(x, y,color = 'red')

plt.plot(x,regressor.predict(x), color = 'blue' )
plt.title('Linear regression for origenal data')
plt.xlabel('equipments')
plt.ylabel('presinor of war')
plt.show()

plt.scatter(x_train, y_train,color = 'red')
plt.scatter(x_test, y_test,color = 'green')
plt.plot(x_train,regressor.predict(x_train), color = 'blue' )
plt.title('Linear regression test and train')
plt.xlabel('equipments')
plt.ylabel('presinor of war')
plt.show()

#################################################################################

##############################   KNN   ################################
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)
from sklearn.model_selection import GridSearchCV
print("\n \n \n")
params = {'n_neighbors':[3,4,5,6,7,8,9]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
print(model.best_params_)


y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print('\n \n \n')
print('The Performance of KNN Regression:',r2,'%')



plt.scatter(x_test, y_predict,color = 'blue')
plt.scatter(x_train,y_train,color = 'red')
plt.title('KNN Regression')
plt.xlabel('equipments')
plt.ylabel('presinor of war')
plt.show()
#####################################################################################


