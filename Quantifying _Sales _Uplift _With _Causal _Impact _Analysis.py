###################################
#Casual impact analysis
###################################

#Casual impact analysis is a time series technique developed by google
#it estimates what would happened known as conterfactual by applying a model to comparable data in a pre-period 
#and projecting this model onto that data in that post -period
#the difference between actual data and the conterfactual in the post-period,
#is the estimated impact of the event


#################################
#Casual impact analysis
#################################


################################
#import required packages
###############################

from causalimpact import CausalImpact
import pandas as pd

#import and create data



#import data tables

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions") 
campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name = "campaign_data") 

#aggregate transactions data to customer data level

customer_daily_sales = transactions.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()



#merge on the signup flag

customer_daily_sales = pd.merge(customer_daily_sales, campaign_data, how = "inner", on = "customer_id")

#pivot the data to aggregate daily sales by signup group

causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                    columns =  "signup_flag",
                                                    values = "sales_cost",
                                                    aggfunc = "mean")

#provide a frequency for our DataFrame index (avoids a warning messaage)

causal_impact_df.index.freq = "D"

#for casual impact we need the impacted group in the first column
causal_impact_df = causal_impact_df[[1,0]]

#rename columns to something  more meaningful

causal_impact_df.columns = ["member", "non_member"]

############################
#Apply Casual Impact
############################
pre_period = ["2020-04-01","2020-06-30"]
post_period = ["2020-07-01", "2020-09-30"]


ci = CausalImpact(causal_impact_df, pre_period, post_period)
##############################
#Plot the Impact
##############################
ci.plot()

##############################
#Extract the summary statistics & report

print(ci.summary())
print(ci.summary(output = "report"))