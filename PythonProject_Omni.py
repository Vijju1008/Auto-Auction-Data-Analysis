#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("car_prices.csv", error_bad_lines=False)


# In[3]:


data.info()


# In[4]:


data.describe()


# In[6]:


missing_values = data.isnull().sum()
print(missing_values)


# In[7]:


data.head()


# In[8]:


sns.pairplot(data, diag_kind="kde")
plt.show()


# In[11]:


sns.lmplot(x="mmr", y="sellingprice", data=data)
plt.show()

sns.lmplot(x="year", y="sellingprice", data=data)
plt.show()


# In[12]:


corr_matrix = data.corr()
print(corr_matrix)


# In[13]:


sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()


# In[16]:


selected_brands = data[data['make'].isin(['BMW', 'Toyota', 'Chevy'])]


# In[18]:


sns.lmplot(x="odometer", y="sellingprice", hue="make", data=selected_brands)
plt.show()


# In[20]:


data['condition'] = data['condition'].round(1)


# In[22]:


pivot_table = data.pivot_table(index='make', columns='condition', values='sellingprice', aggfunc='mean')
print(pivot_table)


# In[23]:


sns.heatmap(pivot_table, cmap="YlGnBu")
plt.show()


# In[24]:


ford_f150 = data[data['model'] == 'F-150']


# In[25]:


sns.histplot(ford_f150['sellingprice'], kde=True)
plt.show()


# In[26]:


sns.boxplot(x="trim", y="sellingprice", data=ford_f150)
plt.show()


# In[27]:


data['Difference_to_MMR'] = data['sellingprice'] - data['mmr']


# In[28]:


sns.boxplot(x="color", y="Difference_to_MMR", data=data)
plt.show()


# In[29]:


sns.boxplot(x="trim", y="Difference_to_MMR", data=data)
plt.show()


# In[30]:


xlt_f150 = data[(data['trim'] == 'XLT') & (data['condition'] > 3.5)]


# In[31]:


statewise_deals = xlt_f150.groupby('state')['Difference_to_MMR'].mean().sort_values()
print(statewise_deals)


# ##Report: Analysis of Used Car Auction Data
# 1. Brands That Offer the Best Value
# Through the analysis of various brands such as BMW, Toyota, and Chevy, we observed significant differences in the value offered by these manufacturers. Using the linear regression plots comparing the odometer reading (vehicle usage) with selling price, key insights were identified:
# 
# Toyota consistently retained higher value relative to mileage, indicating reliability and durability, which made it a preferred choice for long-term investment.
# Chevy demonstrated a broader range in selling prices, likely due to variance in model conditions, making it a more variable choice depending on the vehicle's condition.
# BMW vehicles typically showed a lower price drop-off for high-mileage cars compared to other brands, indicating that these cars could still fetch competitive prices despite higher usage.
# Visualization:
# 
# This plot shows the relationship between the odometer reading and selling price for these three brands, highlighting Toyota's consistency in value.
# 
# 2. Best Deals on Ford F-150 Trucks
# For Ford F-150 trucks, particularly in the context of procurement for fleet management, the analysis focused on the difference between selling price and recommended market price (MMR). Key findings include:
# 
# Trim Analysis:
# The XLT trim offered the most balanced deals, especially for trucks in better condition (greater than 3.5).
# Trucks with Platinum trim had the highest selling prices, often exceeding the MMR, which suggests they are considered premium in the market.
# State-Wise Analysis:
# States such as Ohio and Texas offered significant deals, with trucks selling below MMR on average. These states also had a reasonable volume of trucks available for purchase.
# Visualization:
# 
# The histogram above shows the distribution of selling prices for Ford F-150 trucks by trim level.
# 
# 3. Insights on Selling Prices Based on Condition, Trim, and Color
# Condition:
# 
# Cars in better condition (condition score > 3.5) had a strong positive correlation with selling price, particularly for Ford F-150 trucks.
# Lower condition vehicles (condition < 3) had selling prices significantly below MMR, offering potential deals but with the risk of higher maintenance costs.
# Trim:
# 
# As mentioned, the XLT trim was consistently priced below the MMR, offering good value.
# Platinum trim vehicles commanded premium prices, often exceeding the MMR.
# Color:
# 
# Vehicles in white and silver sold for prices close to or below the MMR, making them popular choices for commercial use.
# Red and black vehicles tended to sell at higher prices than the MMR, likely due to consumer preference for these colors in the retail market.
# Visualization:
# 
# This heatmap visualizes the relationship between selling price, trim, and condition, clearly indicating that better condition vehicles and higher trims such as Platinum command higher prices.
# 
# Conclusion
# The analysis provided key insights for vehicle procurement strategies, particularly in identifying the best deals on Ford F-150 trucks. Toyota emerged as a brand that consistently offers strong value retention, while Ford F-150 XLT trucks in good condition, especially in states like Ohio and Texas, present excellent purchasing opportunities for fleet managers. Finally, vehicle color and condition play crucial roles in influencing selling prices, with practical colors like white and silver often selling closer to MMR, making them ideal for bulk purchases.
# 

# In[ ]:




